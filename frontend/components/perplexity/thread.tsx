"use client";

import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
} from "@assistant-ui/react";
import type { FC } from "react";
import {
  ArrowDownIcon,
  ArrowRightIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  RefreshCwIcon,
  SparkleIcon,
} from "lucide-react";
import Image from "next/image";
import { cn } from "@/lib/utils";

import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  ComposerAttachments,
  UserMessageAttachments,
} from "@/components/assistant-ui/attachment";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root
      className="box-border h-full bg-[#191a1a]"
      style={{
        ["--thread-max-width" as string]: "42rem",
      }}
    >
      <ThreadPrimitive.Empty>
        <ThreadWelcome />
      </ThreadPrimitive.Empty>
      <ThreadPrimitive.If empty={false}>
        {/* Fixed header */}
        <div className="flex items-center justify-between pl-4 pr-4 py-3 absolute top-0 left-0 right-0 z-10 bg-[#191a1a]">
          <Image
            src="/logo.png"
            alt="Queen-RAG"
            width={300}
            height={300}
            className="h-24 w-auto -ml-9"
            priority
          />
        </div>
        <ThreadPrimitive.Viewport className="flex h-full flex-col items-center overflow-y-scroll scroll-smooth bg-inherit px-4 pt-32">
          <ThreadPrimitive.Messages
            components={{
              UserMessage: UserMessage,
              AssistantMessage: AssistantMessage,
            }}
          />

          <div className="min-h-8 flex-grow" />

          <div className="sticky bottom-0 mt-3 flex w-full max-w-[var(--thread-max-width)] flex-col items-center justify-end rounded-t-lg bg-[#191a1a] pb-4">
            <ThreadScrollToBottom />
            <div className="w-full rounded-full bg-foreground/5 p-2">
              <Composer />
            </div>
          </div>
        </ThreadPrimitive.Viewport>
      </ThreadPrimitive.If>
    </ThreadPrimitive.Root>
  );
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="absolute -top-8 rounded-full disabled:invisible"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <div className="flex h-full w-full flex-col">
      {/* Fixed header */}
      <div className="flex items-center justify-between pl-4 pr-4 py-3 absolute top-0 left-0 right-0 z-10 bg-[#191a1a]">
        <Image
          src="/logo.png"
          alt="Queen-RAG"
          width={300}
          height={300}
          className="h-24 w-auto -ml-9"
          priority
        />
      </div>
      <div className="flex h-full w-full items-center justify-center pt-16">
        <div className="flex w-full max-w-[var(--thread-max-width)] flex-grow flex-col gap-12">
          <div className="flex w-full flex-grow flex-col items-center justify-center">
            <p className="font-regular font-display text-2xl md:text-3xl">
              Chat with NEUVIA application documents
            </p>
          </div>
          <ComposerPrimitive.Root className="w-full rounded-lg border bg-[#202222] px-2 shadow-sm transition-all duration-200 outline-none focus-within:ring-1 focus-within:ring-border focus:outline-none">
            <ComposerAttachments />
            <ComposerPrimitive.Input
              rows={1}
              autoFocus
              placeholder="Type your message..."
              className="max-h-40 w-full flex-grow resize-none border-none bg-transparent px-2 py-4 text-lg outline-none placeholder:text-muted-foreground focus:ring-0 disabled:cursor-not-allowed"
            />
            <div className="mx-1.5 flex gap-2">
              <div className="flex-grow" />
              <ComposerPrimitive.Send asChild>
                <TooltipIconButton
                  className="my-2.5 size-8 rounded-full bg-blue-600 hover:bg-blue-700 hover:scale-105 text-white p-2 transition-all"
                  tooltip="Send"
                  variant="ghost"
                >
                  <ArrowRightIcon className="size-4" />
                </TooltipIconButton>
              </ComposerPrimitive.Send>
            </div>
          </ComposerPrimitive.Root>
        </div>
      </div>
    </div>
  );
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="flex w-full flex-wrap items-end rounded-full border border-white/10 bg-[#202222] px-2.5 shadow-sm transition-all duration-200 focus-within:border-white/20">
      <ComposerAttachments />
      <ComposerPrimitive.Input
        rows={1}
        autoFocus
        placeholder="Ask follow-up"
        className="max-h-40 flex-grow resize-none border-none bg-transparent px-4 py-4 text-base outline-none placeholder:text-muted-foreground focus:ring-0 disabled:cursor-not-allowed"
      />
      <div className="flex gap-3">
        <ComposerAction />
      </div>
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <>
      <ThreadPrimitive.If running={false}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip="Send"
            variant="ghost"
            className="my-2.5 size-9 rounded-full bg-blue-600 hover:bg-blue-700 hover:scale-105 text-white p-2 transition-all"
          >
            <ArrowUpIcon className="size-5" />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </ThreadPrimitive.If>
      <ThreadPrimitive.If running>
        <ComposerPrimitive.Cancel asChild>
          <TooltipIconButton
            tooltip="Stop"
            variant="ghost"
            className="my-2.5 size-9 rounded-full bg-red-600 hover:bg-red-700 hover:scale-105 text-white p-2 transition-all"
          >
            <CircleStopIcon />
          </TooltipIconButton>
        </ComposerPrimitive.Cancel>
      </ThreadPrimitive.If>
    </>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative w-full max-w-[var(--thread-max-width)] gap-y-2 py-4">
      <div className="rounded-3xl py-2.5 text-3xl break-words text-foreground">
        <UserMessageAttachments />
        <MessagePrimitive.Parts />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="relative w-full max-w-[var(--thread-max-width)] py-4">
      <div className="my-1.5 leading-7 break-words text-foreground">
        <h1 className="mb-4 inline-flex items-center gap-2 text-2xl">
          <SparkleIcon /> Response
        </h1>

        <MessagePrimitive.Parts components={{ Text: MarkdownText }} />
      </div>

      <div className="flex items-center gap-2 mt-2">
        <BranchPicker />
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="col-start-3 row-start-2 -ml-1 flex gap-1 text-muted-foreground"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy">
          <MessagePrimitive.If copied>
            <CheckIcon />
          </MessagePrimitive.If>
          <MessagePrimitive.If copied={false}>
            <CopyIcon />
          </MessagePrimitive.If>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarPrimitive.Reload asChild>
        <TooltipIconButton tooltip="Refresh">
          <RefreshCwIcon />
        </TooltipIconButton>
      </ActionBarPrimitive.Reload>
    </ActionBarPrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "inline-flex items-center text-xs text-muted-foreground",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};

const CircleStopIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      width="16"
      height="16"
    >
      <rect width="10" height="10" x="3" y="3" rx="2" />
    </svg>
  );
};