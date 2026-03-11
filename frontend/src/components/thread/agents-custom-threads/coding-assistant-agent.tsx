import { v4 as uuidv4 } from "uuid";
import { ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "../messages/ai";
import { HumanMessage } from "../messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { TooltipIconButton } from "../tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  Plus,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "../history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Label } from "../../ui/label";
import { Switch } from "../../ui/switch";
import { GitHubSVG } from "../../icons/github";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../ui/select";
import { useFileUpload } from "@/hooks/use-file-upload";
import { ContentBlocksPreview } from "../ContentBlocksPreview";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "../artifact";
import Link from "next/link";
import { useOpenAiTokenStore } from "@/hooks/use-open-ai-token-store";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div ref={context.contentRef} className={props.contentClassName}>
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="w-4 h-4" />
      <span>Scroll to bottom</span>
    </Button>
  );
}

function OpenGitHubRepo() {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <a
            href="https://github.com/Srednogorie/llm-playground"
            target="_blank"
            className="flex items-center justify-center"
          >
            <GitHubSVG width="24" height="24" />
          </a>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Open GitHub repo</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export function CodingAssistantAgentThread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );

  const {selectedModel, setSelectedModel} = useOpenAiTokenStore();
  const {openAIKey, setOpenAIKey} = useOpenAiTokenStore();

  const [input, setInput] = useState("");
  const {
    contentBlocks,
    setContentBlocks,
    handleFileUpload,
    dropRef,
    removeBlock,
    resetBlocks: _resetBlocks,
    dragOver,
    handlePaste,
  } = useFileUpload();
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const messages = stream.messages;
  const isLoading = stream.isLoading;

  const lastError = useRef<string | undefined>(undefined);
  
  const setThreadId = (id: string | null) => {
    _setThreadId(id);

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const error = stream.error as any;
      const message = error.message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      
      // Check if this is an authentication error
      const isAuthError = 
        error.name === "AuthenticationError" || 
        message.includes("Invalid API Key") ||
        message.includes("401");

      if (isAuthError) {
        toast.error("Authentication Failed", {
          description: (
            <p>
              The API token you provided is invalid or has expired. Please check your token and try again.
            </p>
          ),
          richColors: true,
          closeButton: true,
          duration: 6000,
        });
      } else {
        toast.error("An error occurred. Please try again.", {
          description: (
            <p>
              <strong>Error:</strong> <code>{message}</code>
            </p>
          ),
          richColors: true,
          closeButton: true,
        });
      }
    } catch {
      // no-op
    }
  }, [stream.error]);

  // TODO: this should be part of the useStream hook
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if ((input.trim().length === 0 && contentBlocks.length === 0) || isLoading)
      return;
    setFirstTokenReceived(false);

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: [
        ...(input.trim().length > 0 ? [{ type: "text", text: input }] : []),
        ...contentBlocks,
      ] as Message["content"],
    };

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);
    
    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      { messages: [...toolMessages, newHumanMessage], context },
      {
        streamMode: ["updates"],
        streamSubgraphs: true,
        streamResumable: true,
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
        context: {
          token: openAIKey,
          model: selectedModel,
        }
      },
    );

    setInput("");
    setContentBlocks([]);
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    stream.submit(undefined, {
      checkpoint: parentCheckpoint,
      streamMode: ["updates"],
      streamSubgraphs: true,
      streamResumable: true,
    });
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex w-full h-screen overflow-hidden">
      <div className="relative lg:flex hidden">
        <motion.div
          className="absolute h-full border-r bg-white overflow-hidden z-20"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div className="relative h-full" style={{ width: 300 }}>
            <ThreadHistory />
          </div>
        </motion.div>
      </div>
      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
      <motion.div
        className={cn(
          "flex-1 flex flex-col min-w-0 overflow-hidden relative",
          !chatStarted && "grid-rows-[1fr]",
        )}
        layout={isLargeScreen}
        animate={{
          marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
          width: chatHistoryOpen
            ? isLargeScreen
              ? "calc(100% - 300px)"
              : "100%"
            : "100%",
        }}
        transition={
          isLargeScreen
            ? { type: "spring", stiffness: 300, damping: 30 }
            : { duration: 0 }
        }
      >
        {!chatStarted && (
          <div className="relative top-0 left-0 w-full flex items-center justify-between gap-3 p-2 pl-4 z-10 h-12">
            <div>
              {(!chatHistoryOpen || !isLargeScreen) && (
                <Button
                  className="hover:bg-gray-100"
                  variant="ghost"
                  onClick={() => setChatHistoryOpen((p) => !p)}
                >
                  {chatHistoryOpen ? (
                    <PanelRightOpen className="size-5" />
                  ) : (
                    <PanelRightClose className="size-5" />
                  )}
                </Button>
              )}
            </div>
            <div className="flex items-center justify-center flex-1">
                <Link href="/">Home</Link>
            </div>
            <div className="absolute top-2 right-4 flex items-center">
              <OpenGitHubRepo />
            </div>
          </div>
        )}
        {chatStarted && (
          <div className="flex items-center justify-between gap-3 p-2 z-10 relative">
            <div className="flex items-center justify-start gap-2 relative">
              <div className="absolute left-0 z-10">
                {(!chatHistoryOpen || !isLargeScreen) && (
                  <Button
                    className="hover:bg-gray-100"
                    variant="ghost"
                    onClick={() => setChatHistoryOpen((p) => !p)}
                  >
                    {chatHistoryOpen ? (
                      <PanelRightOpen className="size-5" />
                    ) : (
                      <PanelRightClose className="size-5" />
                    )}
                  </Button>
                )}
              </div>
              <motion.button
                className="flex gap-2 items-center cursor-pointer"
                onClick={() => setThreadId(null)}
                animate={{
                  marginLeft: !chatHistoryOpen ? 48 : 0,
                }}
                transition={{
                  type: "spring",
                  stiffness: 300,
                  damping: 30,
                }}
              >
                <span className="text-xl font-semibold tracking-tight">
                  Coding Assistant
                </span>
              </motion.button>
            </div>
            <div>
              <Link href="/">Home</Link>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center">
                <OpenGitHubRepo />
              </div>
              <TooltipIconButton
                size="lg"
                className="p-4"
                tooltip="New thread"
                variant="ghost"
                onClick={() => setThreadId(null)}
              >
                <SquarePen className="size-5" />
              </TooltipIconButton>
            </div>

            <div className="absolute inset-x-0 top-full h-5" />
          </div>
        )}
          <div className="flex gap-4 h-full overflow-hidden flex-1">
            <StickToBottom className="relative flex-1 overflow-hidden">
              <StickyToBottomContent
                className={cn(
                  "absolute px-4 inset-0 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                  "flex-1 basis-2/3 min-w-0",  // Add these
                  !chatStarted && "flex flex-col items-stretch mt-[25vh]",
                  chatStarted && "grid grid-rows-[1fr_auto]",
                )}
                contentClassName="pt-8 pb-16  max-w-3xl mx-auto flex flex-col gap-4 w-full"
                content={
                  <>
                    {messages
                      .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                      .map((message, index) =>
                        message.type === "human" ? (
                          <HumanMessage
                            key={message.id || `${message.type}-${index}`}
                            message={message}
                            isLoading={isLoading}
                          />
                        ) : (
                          <AssistantMessage
                            key={message.id || `${message.type}-${index}`}
                            message={message}
                            isLoading={isLoading}
                            handleRegenerate={handleRegenerate}
                          />
                        ),
                      )}
                    {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                        We need to render it outside of the messages list, since there are no messages to render */}
                    {hasNoAIOrToolMessages && !!stream.interrupt && (
                      <AssistantMessage
                        key="interrupt-msg"
                        message={undefined}
                        isLoading={isLoading}
                        handleRegenerate={handleRegenerate}
                      />
                    )}
                    {isLoading && !firstTokenReceived && (
                      <AssistantMessageLoading />
                    )}
                  </>
                }
                footer={
                  <div className="sticky flex flex-col items-center gap-8 bottom-0 bg-white">
                    {!chatStarted && (
                      <div className="flex gap-3 items-center">
                        <h1 className="text-2xl font-semibold tracking-tight">
                          Coding Assistant
                        </h1>
                      </div>
                    )}
    
                    <ScrollToBottom className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 animate-in fade-in-0 zoom-in-95" />
    
                    <div
                      ref={dropRef}
                      className={cn(
                        "bg-muted relative z-10 mx-auto mb-8 w-full max-w-3xl rounded-2xl shadow-xs transition-all",
                        dragOver
                          ? "border-primary border-2 border-dotted"
                          : "border border-solid",
                      )}
                    >
                      <form
                        onSubmit={!openAIKey ? e => e.preventDefault() : handleSubmit}
                        className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                      >
                        <ContentBlocksPreview
                          blocks={contentBlocks}
                          onRemove={removeBlock}
                        />
                        <textarea
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                          onPaste={handlePaste}
                          onKeyDown={(e) => {
                            if (
                              e.key === "Enter" &&
                              !e.shiftKey &&
                              !e.metaKey &&
                              !e.nativeEvent.isComposing
                            ) {
                              e.preventDefault();
                              const el = e.target as HTMLElement | undefined;
                              const form = el?.closest("form");
                              form?.requestSubmit();
                            }
                          }}
                          placeholder="Type your message..."
                          className="field-sizing-content resize-none border-none bg-transparent p-3.5 pb-0 shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                        />
  
                        <div className="flex items-center gap-6 p-2 pt-4">
                          <div>
                            <div className="flex items-center space-x-2">
                              <Switch
                                id="render-tool-calls"
                                checked={hideToolCalls ?? false}
                                onCheckedChange={setHideToolCalls}
                              />
                              <Label
                                htmlFor="render-tool-calls"
                                className="text-sm text-gray-600"
                              >
                                Hide Tool Calls
                              </Label>
                            </div>
                          </div>
                          <Label
                            htmlFor="file-input"
                            className="flex cursor-pointer items-center gap-2"
                          >
                            <Plus className="size-5 text-gray-600" />
                            <span className="text-sm text-gray-600">
                              Upload PDF or Image
                            </span>
                          </Label>
                          <input
                            id="file-input"
                            type="file"
                            onChange={handleFileUpload}
                            multiple
                            accept="image/jpeg,image/png,image/gif,image/webp,application/pdf"
                            className="hidden"
                          />
                          {stream.isLoading ? (
                            <Button
                              key="stop"
                              onClick={() => stream.stop()}
                              className="ml-auto"
                            >
                              <LoaderCircle className="h-4 w-4 animate-spin" />
                              Cancel
                            </Button>
                          ) : (
                            <Button
                              type="submit"
                              className="ml-auto shadow-md transition-all"
                              disabled={
                                isLoading ||
                                (!input.trim() && contentBlocks.length === 0) ||
                                !openAIKey
                              }
                            >
                              Send
                            </Button>
                          )}
                        </div>
                      </form>
                    </div>
                  </div>
                }
              />
            </StickToBottom>
            <div className="basis-1/4 overflow-auto  bg-gray-50">
              <div className="p-4">
                <label className="text-sm font-medium mb-2 block">OpenAI API Key</label>
                <div className="flex items-center gap-3">
                  <input
                    type="text"
                    id="token"
                    className="border border-gray-300 rounded-md p-2 w-full bg-white"
                    placeholder="Provide your OpenAI API Key"
                    onChange={(e) => setOpenAIKey(e.target.value)}
                    value={openAIKey || ""}
                  />
                </div>
              </div>
              <div className="p-4">
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Select model</label>
                    <Select value={selectedModel || ""} onValueChange={setSelectedModel}>
                      <SelectTrigger>
                        <SelectValue placeholder="Choose an option..." />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="gpt-5.5">GPT 5.2</SelectItem>
                        <SelectItem value="gpt-5.1">GPT 5.1</SelectItem>
                        <SelectItem value="gpt-5">GPT 5</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative grow" />
          </div>
        </div>
      </div>      
    </div>
  );
}
