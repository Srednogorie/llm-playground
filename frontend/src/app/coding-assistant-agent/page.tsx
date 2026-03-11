"use client";

import { CodingAssistantAgentThread } from "@/components/thread/agents-custom-threads/coding-assistant-agent";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { Toaster } from "@/components/ui/sonner";
import React from "react";
import { ArtifactProvider } from "@/components/thread/artifact";

export default function CodingAssistantAgentPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider currentAssistantId="coding-assistant-agent">
        <StreamProvider currentAssistantId="coding-assistant-agent">
          <ArtifactProvider>
            <CodingAssistantAgentThread />
          </ArtifactProvider>
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
