"use client";

import { useState, useEffect } from "react";
import { Upload, Trash2, FileText, File, AlertCircle, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
// import { cn } from "@/lib/utils";
import { formatBytes } from "@/lib/utils";

interface Document {
  filename: string;
  size: number;
  modified: number;
  type: string;
  metadata?: Record<string, unknown>;
}

interface KnowledgeBaseModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const ITEMS_PER_PAGE = 5;

export function KnowledgeBaseModal({ open, onOpenChange }: KnowledgeBaseModalProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{ current: number; total: number } | null>(null);

  // Fetch documents when modal opens
  useEffect(() => {
    if (open) {
      fetchDocuments();
    }
  }, [open]);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/api/documents/list");
      if (!response.ok) throw new Error("Failed to fetch documents");
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error("Error fetching documents:", error);
      setError("Failed to load documents");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setError(null);
    setUploadProgress({ current: 0, total: files.length });

    let successCount = 0;
    const failedFiles: string[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      setUploadProgress({ current: i + 1, total: files.length });
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/api/documents/upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Upload failed");
        }

        // const result = await response.json();
        successCount++;
      } catch (error) {
        console.error("Upload error:", error);
        failedFiles.push(file.name);
      }
    }

    setUploading(false);
    setUploadProgress(null);

    // Show appropriate toast messages
    if (successCount > 0 && failedFiles.length === 0) {
      toast.success(`Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''}`, {
        position: "bottom-right",
      });
    } else if (successCount > 0 && failedFiles.length > 0) {
      toast.warning(`Uploaded ${successCount} file${successCount > 1 ? 's' : ''}, ${failedFiles.length} failed`, {
        position: "bottom-right",
      });
    } else if (failedFiles.length > 0) {
      toast.error(`Failed to upload: ${failedFiles.join(', ')}`, {
        position: "bottom-right",
      });
    }

    fetchDocuments(); // Refresh the list
  };

  const handleDelete = (filename: string) => {
    setDeletingFile(filename);
  };

  const confirmDelete = async (filename: string) => {
    try {
      const response = await fetch(
        "http://localhost:8000/api/documents/delete",
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ filename }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete document");
      }

      toast.success(`${filename} deleted successfully`, {
        position: "bottom-right",
      });
      fetchDocuments(); // Refresh the list
    } catch (error) {
      console.error("Delete error:", error);
      toast.error(`Failed to delete ${filename}`, {
        position: "bottom-right",
      });
    } finally {
      setDeletingFile(null);
    }
  };

  const cancelDelete = () => {
    setDeletingFile(null);
  };

  const getFileIcon = (type: string) => {
    if (type === "pdf") return <FileText className="h-4 w-4" />;
    return <File className="h-4 w-4" />;
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  // Pagination logic
  const totalPages = Math.ceil(documents.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentDocuments = documents.slice(startIndex, endIndex);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Knowledge Base</DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1">
          {error && (
            <div className="flex items-center gap-2 text-sm text-destructive mb-4">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          {/* Upload Section */}
          {uploading && uploadProgress ? (
            <div className="mb-4 p-6 border-2 border-dashed rounded-lg bg-muted/50">
              <div className="flex flex-col items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary mb-3" />
                <p className="text-sm font-medium mb-1">
                  Uploading {uploadProgress.current} of {uploadProgress.total} files...
                </p>
                <div className="w-full max-w-xs mt-3">
                  <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{
                        width: `${(uploadProgress.current / uploadProgress.total) * 100}%`
                      }}
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Processing and indexing documents...
                </p>
              </div>
            </div>
          ) : (
          <div className="mb-4">
            <label
            htmlFor="dropzone-file"
            className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 border-gray-300 dark:border-gray-600 transition-colors"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <Upload className="w-8 h-8 mb-2 text-gray-400" />
              <p className="mb-2 text-sm text-gray-500 dark:text-gray-400">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                TXT, MD, PDF, DOC, DOCX (MAX. 50MB)
              </p>
            </div>
            <input
              id="dropzone-file"
              type="file"
              multiple
              accept=".txt,.md,.pdf,.doc,.docx"
              className="hidden"
              onChange={handleUpload}
              disabled={uploading}
            />
          </label>
        </div>
          )}

        {/* Documents List */}
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-muted-foreground">
            Documents ({documents.length})
          </h3>

          {loading ? (
            <div className="flex items-center justify-center h-32">
              <span className="text-muted-foreground">Loading documents...</span>
            </div>
          ) : documents.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-center">
              <FileText className="h-6 w-6 text-muted-foreground mb-2" />
              <p className="text-sm text-muted-foreground">No documents in knowledge base</p>
            </div>
          ) : (
            <>
              <div className="space-y-1">
                {currentDocuments.map((doc) => (
                  <div
                    key={doc.filename}
                    className="flex items-center justify-between p-2 rounded border bg-card hover:bg-accent/50 transition-colors"
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {getFileIcon(doc.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{doc.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatBytes(doc.size)} • {formatDate(doc.modified)}
                        </p>
                      </div>
                    </div>
                    {deletingFile === doc.filename ? (
                      <div className="flex items-center gap-1">
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => confirmDelete(doc.filename)}
                          className="h-7 px-3 text-xs font-medium"
                        >
                          Delete
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={cancelDelete}
                          className="h-7 px-3 text-xs font-medium"
                        >
                          Cancel
                        </Button>
                      </div>
                    ) : (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDelete(doc.filename)}
                        className="text-destructive hover:text-destructive h-8 w-8 p-0"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between pt-2">
                  <p className="text-xs text-muted-foreground">
                    Page {currentPage} of {totalPages}
                  </p>
                  <div className="flex gap-1">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(currentPage - 1)}
                      disabled={currentPage === 1}
                      className="h-8 w-8 p-0"
                    >
                      <ChevronLeft className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className="h-8 w-8 p-0"
                    >
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}