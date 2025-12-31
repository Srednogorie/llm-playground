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
        <Link href="/tools-mcp-agent">Tools/MCP Agent</Link>
      </div>
    </div>
    
  );
}
