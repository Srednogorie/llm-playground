"use client";

import { SimpleAgentThread } from "@/components/thread/agents-custom-threads/simple-agent";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { Toaster } from "@/components/ui/sonner";
import React from "react";

export default function DemoPage(): React.ReactNode {
  return (
    <React.Suspense fallback={<div>Loading (layout)...</div>}>
      <Toaster />
      <ThreadProvider currentAssistantId="simple-agent">
        <StreamProvider currentAssistantId="simple-agent">
          <SimpleAgentThread />
        </StreamProvider>
      </ThreadProvider>
    </React.Suspense>
  );
}
