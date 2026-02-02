// app/page.tsx
'use client';

import React, { useState, useCallback } from 'react';
import { Card } from './components/Card';
import { FileIcon } from './components/FileIcon';
import { RuleButton } from './components/RuleButton';
import { SystemFeedback } from './components/SystemFeedback';
import { UploadIcon, FileTextIcon, TrashIcon } from './components/Icons';

interface UploadedFile {
  id: string;
  filename: string;
  filepath: string;
  content?: string;
}

type RuleType = 'temporal' | 'role_completeness' | null;
type ViewMode = 'file' | 'text';

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedFileId, setSelectedFileId] = useState<string | null>(null);
  const [selectedRule, setSelectedRule] = useState<RuleType>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('file');
  const [fileContent, setFileContent] = useState<string>('');
  const [inconsistenciesOutput, setInconsistenciesOutput] = useState<string>('');
  const [progressOutput, setProgressOutput] = useState<string>('');
  const [systemFeedback, setSystemFeedback] = useState<string>('');
  const [pipelineRunning, setPipelineRunning] = useState(false);

  const selectedFile = uploadedFiles.find(f => f.id === selectedFileId);
  const canAnalyze =
    selectedRule !== null &&
    !pipelineRunning &&
    (viewMode === 'text' || selectedFile !== null);

  const showSystemFeedback = useCallback((message: string, duration = 2500) => {
    setSystemFeedback(message);
    setTimeout(() => setSystemFeedback(''), duration);
  }, []);

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;

      Array.from(files).forEach(file => {
        const isDuplicate = uploadedFiles.some(f => f.filepath === file.name);
        if (isDuplicate) {
          showSystemFeedback('This file is already uploaded.');
          return;
        }

        const reader = new FileReader();
        reader.onload = event => {
          const newFile: UploadedFile = {
            id: `${Date.now()}-${Math.random()}`,
            filename: file.name,
            filepath: file.name,
            content: event.target?.result as string
          };
          setUploadedFiles(prev => [...prev, newFile]);
        };
        reader.readAsText(file);
      });
    },
    [uploadedFiles, showSystemFeedback]
  );

  const handleFileSelect = useCallback((fileId: string) => {
    setInconsistenciesOutput('');
    setProgressOutput('');
    setSelectedRule(null);
    setSelectedFileId(fileId);
  }, []);

  const handleLoadFile = useCallback(() => {
    if (!selectedFile?.content) return;
    setFileContent(selectedFile.content);
    setViewMode('text');
  }, [selectedFile]);

  const handleDeleteFile = useCallback(() => {
    if (!selectedFile) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete:\n\n${selectedFile.filename}?`
    );
    if (!confirmed) return;

    setUploadedFiles(prev => prev.filter(f => f.id !== selectedFileId));
    setSelectedFileId(null);
    setInconsistenciesOutput('');
    setProgressOutput('');
    setViewMode('file');
    showSystemFeedback('File removed.');
  }, [selectedFile, selectedFileId, showSystemFeedback]);

  const handleRuleSelect = useCallback(
    (rule: RuleType) => {
      if (pipelineRunning) {
        showSystemFeedback('Analysis in progress. Please wait.');
        return;
      }
      setInconsistenciesOutput('');
      setSelectedRule(rule);
    },
    [pipelineRunning, showSystemFeedback]
  );

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile || !selectedRule) return;

    setPipelineRunning(true);
    setProgressOutput(`Running Temporal Memory Pipeline on:\n${selectedFile.filepath}\n\n`);

    const steps = [
      'Reading Text...',
      'Segmenting Chapters...',
      'Tokenizing Sentences...',
      'Annotating Linguistics...',
      'Loading Sentences...',
      'Constructing Event Frames...',
      'Filling gaps...',
      'Extracting Time Expressions...',
      'Preparing memory for reasoning...',
      'Memory projection complete.',
      'Sending story data to AI for reasoning...',
      'Narrative Validation Complete!!'
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setProgressOutput(prev => prev + step + '\n');
    }

    setInconsistenciesOutput(
      selectedRule === 'temporal'
        ? '<b>Temporal Consistency Analysis:</b><br><br>The story maintains good temporal flow with clear event sequencing. Minor inconsistency detected in Chapter 3 where "the next morning" occurs before the previous evening scene concludes.'
        : '<b>Role Completeness Analysis:</b><br><br>All major characters are properly introduced. The character "Dr. Smith" is mentioned in Chapter 5 without prior introduction or context.'
    );

    setPipelineRunning(false);
  }, [selectedFile, selectedRule]);

  return (
    <div className="max-h-screen bg-[#F6F3FA] p-5 font-['Segoe_UI']">
      <div className="max-w-[1600px] mx-auto flex gap-5 min-h-[650px]">
        {/* Left Panel */}
        <div className="flex-[3] flex flex-col gap-5">
          {systemFeedback && <SystemFeedback message={systemFeedback} />}

          {/* Raw Story Card */}
          <Card title="Raw Story" tint="#F3DDF7">
            <div className="flex gap-2.5 p-2.5">
              <label className="flex items-center gap-2 px-[18px] py-2.5 bg-[#7D5FB5] text-white rounded-md font-medium text-sm cursor-pointer transition-all hover:bg-[#6B4FA0]">
                <UploadIcon />
                Upload Files
                <input
                  type="file"
                  multiple
                  accept=".txt"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <button
                className="flex items-center gap-2 px-[18px] py-2.5 bg-[#E8E1F5] text-[#5A4A7A] rounded-md font-medium text-sm transition-all hover:bg-[#D5C8F0] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleLoadFile}
                disabled={!selectedFile}
              >
                <FileTextIcon />
                Select File
              </button>
              <button
                className="flex items-center gap-2 px-[18px] py-2.5 bg-[#FFE5E5] text-[#C44444] rounded-md font-medium text-sm transition-all hover:bg-[#FFD0D0] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleDeleteFile}
                disabled={!selectedFile}
              >
                <TrashIcon />
                Delete File
              </button>
            </div>

            {/* File Display */}
            <div className="flex-1 min-h-[230px] mx-3 mb-3">
              {viewMode === 'file' ? (
                <div className="h-full bg-[#FAF8FE] border-2 border-dashed border-[#D7CFF1] rounded-lg p-5 overflow-y-auto">
                  {uploadedFiles.length === 0 ? (
                    <div className="flex items-center justify-center h-full text-[#9A90B8] text-center leading-relaxed">
                      Drag and drop your story file/s here<br />(.txt files)
                    </div>
                  ) : (
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(110px,1fr))] gap-3 p-2.5">
                      {uploadedFiles.map(file => (
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
                  checked={viewMode === 'file'}
                  onChange={() => setViewMode('file')}
                  className="cursor-pointer"
                />
                File Mode
              </label>
              <label className="flex items-center gap-1.5 cursor-pointer text-sm text-[#5A4A7A]">
                <input
                  type="radio"
                  name="viewMode"
                  checked={viewMode === 'text'}
                  onChange={() => setViewMode('text')}
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
                selected={selectedRule === 'temporal'}
                disabled={!selectedFile}
                onClick={() => handleRuleSelect('temporal')}
              />
              <RuleButton
                title="Role Completeness 🎭"
                description="Checks that all important characters and tools are present when something happens in the story."
                selected={selectedRule === 'role_completeness'}
                disabled={!selectedFile}
                onClick={() => handleRuleSelect('role_completeness')}
              />
            </div>
          </Card>
        </div>

        {/* Right Panel */}
        <div className="flex-[2] flex flex-col gap-5">
          <Card title="Memo Weave Feedback" tint="#FFE2E2">
            <div
              className="p-3 min-h-[200px] max-h-[25dvh] overflow-y-auto border border-[#E0DAF0] bg-white rounded-md text-sm leading-relaxed text-[#2D2640] whitespace-pre-wrap break-words"
              dangerouslySetInnerHTML={{
                __html:
                  inconsistenciesOutput ||
                  '<span class="text-[#9A90B8]">Detail inconsistencies, story-rule conflicts, and other violations flagged will appear here...</span>'
              }}
            />
          </Card>

          <Card title="Memo Weave System Progress" tint="#EFEFEF">
            <div className="p-3 min-h-[200px] max-h-[25dvh] overflow-y-auto border border-[#E0DAF0] bg-white rounded-md text-sm leading-relaxed text-[#2D2640] whitespace-pre-wrap break-words">
              {progressOutput || (
                <span className="text-[#9A90B8]">
                  Reading Text...<br />
                  Segmenting Chapters...<br />
                  Tokenizing Sentences...<br />
                  Annotating Linguistics...<br />
                  Loading Sentences...<br />
                  Constructing Event Frames...<br />
                  Filling gaps...<br />
                  Extracting Time Expressions...
                </span>
              )}
            </div>
          </Card>

          <button
            className="h-[50px] px-6 py-3 bg-gradient-to-r from-[rgba(125,95,181,0.75)] to-[rgba(199,132,255,1)] text-white rounded-xl text-lg font-semibold transition-all self-end hover:from-[rgba(138,106,209,0.85)] hover:to-[rgba(199,132,255,1)] hover:-translate-y-0.5 hover:shadow-[0_4px_12px_rgba(125,95,181,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleAnalyze}
            disabled={!canAnalyze}
          >
<<<<<<< HEAD
            ✨ Analyze by Memo Weave
=======
            ✨ Analyze with MemoWeave AI
>>>>>>> master
          </button>
        </div>
      </div>
    </div>
  );
}
