export type ReadingMode = "original" | "ai_natural";

export type BlockType = "heading" | "paragraph" | "list" | "quote" | "footnote";

export interface SpeechChunk {
  id: string;
  docId: string;
  chunkIndex: number;
  pageNumber: number;
  chapterTitle: string;
  blockType: BlockType;
  originalText: string;
  cleanedText: string;
  charStart: number;
  charEnd: number;
  estimatedDurationSec: number;
}

export interface ReadingMemoryState {
  docId: string;
  currentPage: number;
  currentParagraph: number;
  currentSentence: number;
  currentChunkIndex: number;
  charOffset: number;
  playbackSpeed: number;
  selectedVoice: string;
  readingMode: ReadingMode;
  lastTimestamp: number;
  completionPct: number;
}

export interface PdfDocumentRecord {
  id: string;
  title: string;
  author: string;
  filename: string;
  pageCount: number;
  totalChunks: number;
  fileSize: number;
  coverColor: string;
  createdAt: number;
  lastOpenedAt: number;
  chunks: SpeechChunk[];
  chapters: { title: string; startChunkIndex: number; pageNumber: number }[];
  summary: string;
  keyConcepts: { term: string; definition: string; page: number }[];
}
