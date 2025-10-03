declare module 'ai/react' {
  export interface Message {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
  }

  export interface UseChatOptions {
    api?: string;
    body?: Record<string, unknown>;
    headers?: Record<string, string>;
    onError?: (error: Error) => void;
  }

  export interface UseChatHelpers {
    messages: Message[];
    input: string;
    handleInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
    isLoading: boolean;
    error: Error | null;
    reload: () => void;
    stop: () => void;
    append: (message: Message) => void;
    setMessages: (messages: Message[]) => void;
  }

  export function useChat(options?: UseChatOptions): UseChatHelpers;
}