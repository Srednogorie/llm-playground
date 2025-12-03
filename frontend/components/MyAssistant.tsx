"use client";
import {
  AssistantRuntimeProvider,
  ChatModelAdapter,
  useLocalRuntime,
} from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread";
import { useSettings } from "@/lib/SettingsContext";

const MyModelAdapter = (temperature: number, max_tokens: number): ChatModelAdapter => ({
  async run({ messages, abortSignal }) {
    const req_body = {
      assistant_id: "agent",
      input: {
        messages: messages.map((message) => ({
          role: message.role,
          content: message.content[0].text,
        })),
      },
      context: {temperature, max_tokens},
    };
    console.log(req_body);
    // TODO replace with your own API
    const result = await fetch("http://127.0.0.1:8123/runs/wait", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // forward the messages in the chat to the API
      body: JSON.stringify(req_body),
      // if the user hits the "cancel" button or escape keyboard key, cancel the request
      signal: abortSignal,
    });
    const data = await result.json();
    console.log(data);
    return {
      content: [
        {
          type: "text",
          text: data.messages[data.messages.length - 1].content,
        },
      ],
      metadata: {
        custom: {
          input_tokens: data.messages[data.messages.length - 1].usage_metadata.input_tokens,
          output_tokens: data.messages[data.messages.length - 1].usage_metadata.output_tokens,
          total_tokens: data.messages[data.messages.length - 1].usage_metadata.total_tokens,
        },
      },
    };
  },
});

export function MyAssistant() {
  const { settings } = useSettings();
  const runtime = useLocalRuntime(MyModelAdapter(settings.temperature, settings.max_tokens))
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  );
}
