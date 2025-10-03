"use client";

import { FC, memo } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { cn } from "@/lib/utils";

export type MarkdownTextProps = {
  text: string;
  className?: string;
};

export const MarkdownText: FC<MarkdownTextProps> = memo(({ text, className }) => {
  const components: Partial<Components> = {
    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
    ul: ({ children }) => <ul className="mb-2 ml-4 list-disc">{children}</ul>,
    ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal">{children}</ol>,
    li: ({ children }) => <li className="mb-1">{children}</li>,
    h1: ({ children }) => <h1 className="mb-4 text-2xl font-bold">{children}</h1>,
    h2: ({ children }) => <h2 className="mb-3 text-xl font-bold">{children}</h2>,
    h3: ({ children }) => <h3 className="mb-2 text-lg font-semibold">{children}</h3>,
    h4: ({ children }) => <h4 className="mb-2 font-semibold">{children}</h4>,
    h5: ({ children }) => <h5 className="mb-1 font-medium">{children}</h5>,
    h6: ({ children }) => <h6 className="mb-1 text-sm font-medium">{children}</h6>,
    blockquote: ({ children }) => (
      <blockquote className="mb-2 border-l-4 border-gray-300 pl-4 italic">
        {children}
      </blockquote>
    ),
    code: ({ className, children, ...props }) => {
      const match = /language-(\w+)/.exec(className || "");
      return match ? (
        <pre className="mb-2 overflow-x-auto rounded-md bg-gray-100 p-4 dark:bg-gray-900">
          <code className={className} {...props}>
            {children}
          </code>
        </pre>
      ) : (
        <code
          className="rounded-md bg-gray-100 px-1.5 py-0.5 dark:bg-gray-900"
          {...props}
        >
          {children}
        </code>
      );
    },
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 hover:underline dark:text-blue-400"
      >
        {children}
      </a>
    ),
    hr: () => <hr className="my-4 border-t border-gray-300" />,
    table: ({ children }) => (
      <table className="mb-2 border-collapse border border-gray-300">
        {children}
      </table>
    ),
    thead: ({ children }) => <thead className="bg-gray-50">{children}</thead>,
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => <tr className="border-b border-gray-300">{children}</tr>,
    th: ({ children }) => (
      <th className="border border-gray-300 px-4 py-2 text-left font-semibold">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border border-gray-300 px-4 py-2">{children}</td>
    ),
  };

  return (
    <div className={cn("prose prose-sm max-w-none dark:prose-invert", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});

MarkdownText.displayName = "MarkdownText";