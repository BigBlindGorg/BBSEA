'use client';

import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
  type ChatModelRunOptions,
  type AttachmentAdapter,
  INTERNAL,
} from '@assistant-ui/react';
import { type ReactNode } from 'react';

const { generateId } = INTERNAL;

// Extended types for our attachment handling
interface ExtendedAttachment {
  id: string;
  type: 'image' | 'document' | 'file';
  name: string;
  file: File;
  status: { type: 'pending' | 'complete' };
  url?: string; // For image data URLs
  contentType?: string;
  content?: Array<{
    type: 'image' | 'file';
    image?: string;
    data?: string;
    filename?: string;
  }>;
}

interface ExtendedMessagePart {
  type: string;
  text?: string;
  image?: string;
  data?: string;
  mimeType?: string;
  mediaType?: string;
  filename?: string;
  name?: string;
  file?: File;
}

const RAGAdapter: ChatModelAdapter = {
  async *run(options: ChatModelRunOptions) {
    // Extract the user's message
    const messages = options.messages;
    const lastMessage = messages[messages.length - 1];

    if (!lastMessage || lastMessage.role !== 'user') {
      throw new Error('No user message found');
    }

    // Process attachments for vision model
    const attachments: Array<{ type: 'image' | 'document'; content: string; name: string }> = [];

    // Helper function to convert file to base64
    const getFileDataURL = async (file: File): Promise<string> => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    };

    // Check attachments array (Assistant UI 0.6+ format)
    if ('attachments' in lastMessage && Array.isArray(lastMessage.attachments)) {
      for (const attachment of lastMessage.attachments) {
        const extAttachment = attachment as unknown as ExtendedAttachment;
        try {
          // Handle image attachments for vision
          if (extAttachment.type === 'image') {
            // Check for content array (this is how AttachmentAdapter sends completed attachments)
            if (extAttachment.content && Array.isArray(extAttachment.content)) {
              for (const contentPart of extAttachment.content) {
                if (contentPart.type === 'image' && contentPart.image) {
                  attachments.push({
                    type: 'image',
                    content: contentPart.image, // This is the base64 data URL
                    name: extAttachment.name || contentPart.filename || 'image'
                  });
                }
              }
            }
            // Check for url (base64 data URL from our AttachmentAdapter)
            else if (extAttachment.url) {
              attachments.push({
                type: 'image',
                content: extAttachment.url,
                name: extAttachment.name || 'image'
              });
            }
            // Check for file object
            else if (extAttachment.file instanceof File) {
              const dataUrl = await getFileDataURL(extAttachment.file);
              attachments.push({
                type: 'image',
                content: dataUrl,
                name: extAttachment.file.name
              });
            }
          }
          // Handle document attachments
          else if (extAttachment.type === 'document' || extAttachment.type === 'file') {
            // Check for content array
            if (extAttachment.content && Array.isArray(extAttachment.content)) {
              for (const contentPart of extAttachment.content) {
                if (contentPart.type === 'file' && contentPart.data) {
                  attachments.push({
                    type: 'document',
                    content: contentPart.data,
                    name: extAttachment.name || contentPart.filename || 'document'
                  });
                }
              }
            }
            // Check for file object
            else if (extAttachment.file instanceof File) {
              const content = await extAttachment.file.text();
              attachments.push({
                type: 'document',
                content: content,
                name: extAttachment.file.name
              });
            }
          }
        } catch (error) {
          console.error('Failed to process attachment:', error);
        }
      }
    }

    // Also check content parts for backwards compatibility
    for (const part of lastMessage.content) {
      const extPart = part as ExtendedMessagePart;
      console.log('Part type:', extPart.type, 'Part:', extPart);

      // Handle image attachments for vision
      if (extPart.type === 'image' && extPart.image) {
        attachments.push({
          type: 'image',
          content: extPart.image, // Base64 data URL
          name: 'image.png'
        });
      }
      // Handle file attachments - Assistant UI uses 'file' type with data property
      else if (extPart.type === 'file') {
        try {
          // Check for data property (base64 encoded file)
          if (extPart.data) {
            const mimeType = extPart.mimeType || extPart.mediaType || '';
            if (mimeType.startsWith('image/')) {
              attachments.push({
                type: 'image',
                content: extPart.data, // Already base64 data URL
                name: extPart.filename || extPart.name || 'image'
              });
            } else {
              // For text files, the data might be base64 encoded
              attachments.push({
                type: 'document',
                content: extPart.data,
                name: extPart.filename || extPart.name || 'document'
              });
            }
          }
          // Fallback: check for file property (File object)
          else if (extPart.file instanceof File) {
            const file = extPart.file;
            if (file.type.startsWith('image/')) {
              // Convert image to base64 for vision
              const dataUrl = await getFileDataURL(file);
              attachments.push({
                type: 'image',
                content: dataUrl,
                name: file.name
              });
            } else {
              // For documents, read as text
              const content = await file.text();
              attachments.push({
                type: 'document',
                content: content,
                name: file.name
              });
            }
          }
        } catch (error) {
          console.error('Failed to process file part:', error);
        }
      }
    }

    // Extract text from message
    const textParts = lastMessage.content
      .filter((part): part is { type: 'text'; text: string } =>
        part.type === 'text' && typeof part.text === 'string'
      )
      .map(part => part.text);

    const userMessage = textParts.join(' ');

    console.log('Sending message with attachments:', {
      message: userMessage,
      attachmentCount: attachments.length,
      attachmentTypes: attachments.map(a => a.type)
    });

    // Prepare request body with attachments for vision and documents
    interface ChatRequestBody {
      message: string;
      history: Array<{ role: string; content: string }>;
      use_rag: boolean;
      stream: boolean;
      attachments?: Array<{ type: string; name: string; content: string }>;
    }

    const requestBody: ChatRequestBody = {
      message: userMessage,
      history: messages
        .slice(0, -1)
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .map(msg => ({
          role: msg.role,
          content: msg.role === 'user'
            ? msg.content.filter(p => p.type === 'text').map((p: ExtendedMessagePart) => p.text).join(' ')
            : msg.content.filter((p: ExtendedMessagePart) => p.type === 'text').map((p: ExtendedMessagePart) => p.text).join(' ')
        })),
      use_rag: true,
      stream: true
    };

    // Add attachments (images and documents) for processing
    if (attachments.length > 0) {
      requestBody.attachments = attachments.map(a => ({
        type: a.type,
        name: a.name,
        content: a.content // Base64 data URL for images, text content for documents
      }));

      console.log(`Sending ${attachments.length} attachment(s):`, {
        types: attachments.map(a => a.type),
        names: attachments.map(a => a.name)
      });
    }

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';
      const response = await fetch(`${apiUrl}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`RAG API error: ${response.status} - ${errorText}`);
      }

      // Check if response is actually streaming
      const contentType = response.headers.get('content-type');
      if (!contentType?.includes('text/event-stream')) {
        const result = await response.json();
        yield {
          content: [{ type: 'text', text: result.response || result.message || 'No response' }],
          role: 'assistant' as const,
        };
        return;
      }

      // Handle streaming response
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              // Yield final accumulated message if there's any remaining text
              if (accumulatedText) {
                yield {
                  content: [{ type: 'text', text: accumulatedText }],
                  role: 'assistant' as const,
                };
              }
              return;
            }
            try {
              const parsed = JSON.parse(data);
              // Handle both token format (old) and content format (new backend)
              if (parsed.token) {
                accumulatedText += parsed.token;
                yield {
                  content: [{ type: 'text', text: accumulatedText }],
                  role: 'assistant' as const,
                };
              } else if (parsed.type === 'content' && parsed.content) {
                accumulatedText += parsed.content;
                yield {
                  content: [{ type: 'text', text: accumulatedText }],
                  role: 'assistant' as const,
                };
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('RAG API Error:', error);
      yield {
        content: [{
          type: 'text',
          text: `Error: ${error instanceof Error ? error.message : 'Failed to get response from RAG API'}`
        }],
        role: 'assistant' as const,
      };
    }
  },
};

// Helper function to convert file to base64
const getFileDataURL = async (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

// Attachment adapter to handle file uploads with vision support
const attachmentAdapter: AttachmentAdapter = {
  accept: 'text/*,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/*',

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async add({ file }: { file: File }): Promise<any> {
    const isImage = file.type.startsWith('image/');

    // For images, prepare for vision model
    if (isImage) {
      const dataUrl = await getFileDataURL(file);
      return {
        id: generateId(),
        type: 'image' as const,
        name: file.name,
        contentType: file.type,
        file,
        url: dataUrl, // Base64 data URL for vision
        status: { type: 'pending' as const }
      };
    }

    // For documents
    return {
      id: generateId(),
      type: 'document' as const,
      name: file.name,
      contentType: file.type,
      file,
      status: { type: 'pending' as const }
    };
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async send(attachment: any): Promise<any> {
    // Complete the attachment with content array as per Assistant UI spec
    if (attachment.type === 'image' && attachment.url) {
      return {
        ...attachment,
        status: { type: 'complete' as const },
        content: [{
          type: 'image' as const,
          image: attachment.url // The base64 data URL
        }]
      };
    }

    // For documents, include the text content
    if ((attachment.type === 'document' || attachment.type === 'file') && attachment.file) {
      const text = await attachment.file.text();
      return {
        ...attachment,
        status: { type: 'complete' as const },
        content: [{
          type: 'file' as const,
          data: text,
          // mimeType is not in the current type definition
          // but may be needed for future compatibility
        }]
      };
    }

    // Default: just mark as complete
    return {
      ...attachment,
      status: { type: 'complete' as const }
    };
  },

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async remove(attachment: any) {
    // Handle removal if needed
    console.log('Removing attachment:', attachment.id);
  }
};

export function AssistantProvider({ children }: { children: ReactNode }) {
  const runtime = useLocalRuntime(RAGAdapter, {
    adapters: {
      attachments: attachmentAdapter
    }
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}