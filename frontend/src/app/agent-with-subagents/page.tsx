"use client";

import { AgentWithSubagentsThread } from "@/components/thread/agents-custom-threads/agent-with-subagents";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { Toaster } from "@/components/ui/sonner";
import React from "react";
import { ArtifactProvider } from "@/components/thread/artifact";

export default function DeepAgentTestPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider currentAssistantId="agent-with-subagents">
        <StreamProvider currentAssistantId="agent-with-subagents">
          <ArtifactProvider>
            <AgentWithSubagentsThread />
          </ArtifactProvider>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
