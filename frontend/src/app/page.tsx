"use client";

import React from "react";
import Link from "next/link";

export default function DemoPage(): React.ReactNode {
  return (
    <div>
      <div>
        <Link href="/simple-agent">Simple Agent</Link>
      </div>
      <div>
        <Link href="/tools-mcp-agent">Tools/MCP/Skills Agent</Link>
      </div>
      <div>
        <Link href="/agent-with-subagents">Agent With Subagents</Link>
      </div>
      <div>
        <Link href="/coding-assistant-agent">Coding Assistant Agent With Human-In-The-Loop</Link>
      </div>
    </div>
  );
}
