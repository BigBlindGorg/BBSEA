"use client";

import { useState } from "react";
import { Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { KnowledgeBaseModal } from "./knowledge-base-modal";

export function KnowledgeBaseButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setOpen(true)}
        className="flex items-center gap-2"
      >
        <Database className="h-4 w-4" />
        <span className="hidden sm:inline">Knowledge Base</span>
      </Button>
      <KnowledgeBaseModal open={open} onOpenChange={setOpen} />
    </>
  );
}