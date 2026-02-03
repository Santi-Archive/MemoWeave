// app/page.tsx
"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { Card } from "./components/Card";
import { FileIcon } from "./components/FileIcon";
import { RuleButton } from "./components/RuleButton";
import { SystemFeedback } from "./components/SystemFeedback";
import { UploadIcon, FileTextIcon, TrashIcon } from "./components/Icons";

// wire to backend url or localhost
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface UploadedFile {
  id: string;
  filename: string;
  filepath: string;
  content?: string;
}

type RuleType = "temporal" | "role_completeness" | null;
type ViewMode = "file" | "text";

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [selectedRule, setSelectedRule] = useState<RuleType>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("file");
  const [fileContent, setFileContent] = useState<string>("");
  const [inconsistenciesOutput, setInconsistenciesOutput] =
    useState<string>("");
  const [progressOutput, setProgressOutput] = useState<string>("");
  const [systemFeedback, setSystemFeedback] = useState<string>("");
  const [pipelineRunning, setPipelineRunning] = useState(false);

  // Ref for EventSource to close it if component unmounts
  const eventSourceRef = useRef<EventSource | null>(null);
  const isInitialMount = useRef(true);

  const selectedFile = uploadedFiles.find((f) => f.id === selectedFileId);
  const canAnalyze =
    selectedRule !== null &&
    !pipelineRunning &&
    (viewMode === "text" || selectedFile !== null);

  const showSystemFeedback = useCallback((message: string, duration = 2500) => {
    setSystemFeedback(message);
    setTimeout(() => setSystemFeedback(""), duration);
  }, []);

  // Fetch files on mount
  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/files`);

      if (!res.ok) {
        // Backend responded, but with error
        if (!isInitialMount.current) {
          console.error("Failed to fetch files:", res.statusText);
        }
        return;
      }

      const files = await res.json();
      setUploadedFiles(
        files.map((f: any) => ({
          id: f.filename,
          filename: f.filename,
          filepath: f.filepath,
        })),
      );
    } catch (err) {
      // Network / connection error
      if (!isInitialMount.current) {
        console.error("Failed to fetch files", err);
      }
      // silently ignore on first load
    } finally {
      isInitialMount.current = false;
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  const handleFileUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;

      for (const file of Array.from(files)) {
        if (uploadedFiles.some((f) => f.filename === file.name)) {
          showSystemFeedback(`File ${file.name} already uploaded.`);
          continue;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
          const res = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData,
          });

          if (res.ok) {
            showSystemFeedback(`Uploaded ${file.name}`);
            fetchFiles(); // Refresh list
          } else {
            showSystemFeedback(`Failed to upload ${file.name}`);
          }
        } catch (err) {
          console.error(err);
          showSystemFeedback(`Error uploading ${file.name}`);
        }
      }
    },
    [uploadedFiles, showSystemFeedback, fetchFiles],
  );

  const handleFileSelect = useCallback((fileId: string) => {
    if (pipelineRunning) {
      showSystemFeedback("Analysis in progress. Please wait.");
      return;
    }
    setInconsistenciesOutput("");
    setProgressOutput("");
    setSelectedRule(null);
    setSelectedFileId(fileId);
  }, [pipelineRunning, showSystemFeedback]);

  // Fetch content when user wants to view text
  const handleLoadFile = useCallback(async () => {
    if (!selectedFile) return;

    try {
      const res = await fetch(
        `${API_URL}/files/${selectedFile.filename}/content`,
      );
      if (res.ok) {
        const data = await res.json();
        setFileContent(data.content);
        setViewMode("text");
      } else {
        showSystemFeedback("Could not load file content.");
      }
    } catch (err) {
      showSystemFeedback("Error loading file content.");
    }
  }, [selectedFile, showSystemFeedback]);

  const handleDeleteFile = useCallback(async () => {
    if (!selectedFile) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete:\n\n${selectedFile.filename}?`,
    );
    if (!confirmed) return;

    try {
      const res = await fetch(`${API_URL}/files/${selectedFile.filename}`, {
        method: "DELETE",
      });

      if (res.ok) {
        setUploadedFiles((prev) => prev.filter((f) => f.id !== selectedFileId));
        setSelectedFileId(null);
        setInconsistenciesOutput("");
        setProgressOutput("");
        setViewMode("file");
        showSystemFeedback("File removed.");
      } else {
        showSystemFeedback("Failed to delete file.");
      }
    } catch (err) {
      showSystemFeedback("Error deleting file.");
    }
  }, [selectedFile, selectedFileId, showSystemFeedback]);

  const handleRuleSelect = useCallback(
    (rule: RuleType) => {
      if (pipelineRunning) {
        showSystemFeedback("Analysis in progress. Please wait.");
        return;
      }
      setInconsistenciesOutput("");
      setSelectedRule(rule);
    },
    [pipelineRunning, showSystemFeedback],
  );

  const handleAnalyze = useCallback(() => {
    if (!selectedFile || !selectedRule) return;

    setPipelineRunning(true);
    setProgressOutput(
      `Initializing analysis for ${selectedFile.filename}...\n`,
    );
    setInconsistenciesOutput("");

    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const url = `${API_URL}/analyze_stream?filename=${encodeURIComponent(selectedFile.filename)}&rule=${selectedRule}`;
    const evtSource = new EventSource(url);
    eventSourceRef.current = evtSource;

    evtSource.onopen = () => {
      console.log("SSE connection opened");
    };

    evtSource.onerror = (e) => {
      console.error("SSE error", e);
    };

    evtSource.onmessage = (event) => {
      // Append log to progress
      setProgressOutput((prev) => prev + event.data + "\n");
    };

    evtSource.addEventListener("result", (event: MessageEvent) => {
      const data = JSON.parse(event.data);
      setInconsistenciesOutput(data.feedback);
      evtSource.close();
      setPipelineRunning(false);
    });

    evtSource.addEventListener("error", (event: MessageEvent) => {
      if (eventSourceRef.current?.readyState === EventSource.CLOSED) {
        setPipelineRunning(false);
      } else {
        // connection error
        // setProgressOutput(prev => prev + "[Connection interrupted]\n");
        evtSource.close();
        setPipelineRunning(false);
      }
    });

    // Custom error event from server
    evtSource.addEventListener("error_msg", (event: MessageEvent) => {
      setProgressOutput((prev) => prev + "[ERROR] " + event.data + "\n");
      evtSource.close();
      setPipelineRunning(false);
    });
  }, [selectedFile, selectedRule]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#F6F3FA] p-4 sm:p-5 font-['Segoe_UI']">
      <div className="max-w-[1600px] mx-auto flex flex-col lg:flex-row gap-4 sm:gap-5 min-h-[550px]">
        {/* Left Panel */}
        <div className="flex-[3] flex flex-col gap-4 sm:gap-5">
          {systemFeedback && <SystemFeedback message={systemFeedback} />}

          {/* Raw Story Card */}
          <Card title="Raw Story" tint="#F3DDF7">
            <div className="flex gap-2.5 p-2.5">
              <label className={`flex items-center gap-2 px-[18px] py-2.5 rounded-md font-medium text-sm transition-all ${
                pipelineRunning
                  ? 'bg-[#B5A5D5] text-white cursor-not-allowed opacity-50'
                  : 'bg-[#7D5FB5] text-white cursor-pointer hover:bg-[#6B4FA0]'
              }`}>
                <UploadIcon />
                Upload Files
                <input
                  type="file"
                  multiple
                  accept=".txt"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={pipelineRunning}
                />
              </label>
              <button
                className="flex items-center gap-2 px-[18px] py-2.5 bg-[#E8E1F5] text-[#5A4A7A] rounded-md font-medium text-sm transition-all hover:bg-[#D5C8F0] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleLoadFile}
                disabled={!selectedFile || pipelineRunning}
              >
                <FileTextIcon />
                Select File
              </button>
              <button
                className="flex items-center gap-2 px-[18px] py-2.5 bg-[#FFE5E5] text-[#C44444] rounded-md font-medium text-sm transition-all hover:bg-[#FFD0D0] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleDeleteFile}
                disabled={!selectedFile || pipelineRunning}
              >
                <TrashIcon />
                Delete File
              </button>
            </div>

            {/* File Display */}
            <div className="flex-1 min-h-[200px] mx-2.5 mb-2.5">
              {viewMode === "file" ? (
                <div className="h-full bg-[#FAF8FE] border-2 border-dashed border-[#D7CFF1] rounded-lg p-5 overflow-y-auto">
                  {uploadedFiles.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-[#9A90B8] text-center leading-relaxed">
                      Drag and drop your story file/s here
                      <br />
                      (.txt files)
                    </div>
                  ) : (
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(95px,1fr))] gap-5 p-2">
                      {uploadedFiles.map((file) => (
                        <FileIcon
                          key={file.id}
                          filename={file.filename}
                          filepath={file.filepath}
                          selected={file.id === selectedFileId}
                          onClick={() => handleFileSelect(file.id)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <textarea
                  className="w-full h-full p-3 border border-[#E0DAF0] rounded-md font-['Consolas','Courier_New',monospace] text-[13px] resize-none bg-white"
                  value={fileContent}
                  readOnly
                />
              )}
            </div>

            {/* View Mode Switch */}
            <div className="flex justify-center gap-5 px-3 pb-2.5 mb-2.5">
              <label className="flex items-center gap-1.5 cursor-pointer text-sm text-[#5A4A7A]">
                <input
                  type="radio"
                  name="viewMode"
                  checked={viewMode === "file"}
                  onChange={() => setViewMode("file")}
                  className="cursor-pointer"
                />
                File Mode
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-sm text-[#5A4A7A]">
                <input
                  type="radio"
                  name="viewMode"
                  checked={viewMode === "text"}
                  onChange={() => setViewMode("text")}
                  disabled={!selectedFile}
                  className="cursor-pointer disabled:cursor-not-allowed"
                />
                Text Mode
              </label>
            </div>
          </Card>

          {/* Story Rules Card */}
          <Card title="Story Consistency Rules" tint="#DDE8FF">
            <div className="grid grid-cols-2 gap-3.5 p-3">
              <RuleButton
                title="Temporal Consistency ⏱️"
                description="Checks that events happen in a logical order and at the right times. Detects contradictions in timing or overlapping events."
                selected={selectedRule === "temporal"}
                disabled={!selectedFile}
                onClick={() => handleRuleSelect("temporal")}
              />
              <RuleButton
                title="Role Completeness 🎭"
                description="Checks that all important characters and tools are present when something happens in the story."
                selected={selectedRule === "role_completeness"}
                disabled={!selectedFile}
                onClick={() => handleRuleSelect("role_completeness")}
              />
            </div>
          </Card>
        </div>

        {/* Right Panel */}
        <div className="flex-[2] flex flex-col gap-4 sm:gap-5">
          <Card title="Memo Weave Feedback" tint="#FFE2E2">
            <div
              className="p-3 min-h-[180px] max-h-[28dvh] overflow-y-auto border border-[#E0DAF0] bg-white rounded-md text-sm leading-relaxed text-[#2D2640] whitespace-pre-wrap break-words"
              dangerouslySetInnerHTML={{
                __html:
                  inconsistenciesOutput ||
                  '<span class="text-[#9A90B8]">Detail inconsistencies, story-rule conflicts, and other violations flagged will appear here...</span>',
              }}
            />
          </Card>

          <Card title="Memo Weave System Progress" tint="#EFEFEF">
            <div className="p-3 min-h-[180px] max-h-[28dvh] overflow-y-auto border border-[#E0DAF0] bg-white rounded-md text-sm leading-relaxed text-[#2D2640] whitespace-pre-wrap break-words">
              {progressOutput || (
                <span className="text-[#9A90B8]">
                  System logs will appear here during analysis...
                </span>
              )}
            </div>
          </Card>

          <button
            className="h-[48px] px-5 py-2.5 bg-gradient-to-r from-[rgba(125,95,181,0.75)] to-[rgba(199,132,255,1)] text-white rounded-xl text-base sm:text-lg font-semibold transition-all self-end lg:self-end hover:from-[rgba(138,106,209,0.85)] hover:to-[rgba(199,132,255,1)] hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(125,95,181,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleAnalyze}
            disabled={!canAnalyze}
          >
            ✨ Analyze with MemoWeave AI
          </button>
        </div>
      </div>
    </div>
  );
}
