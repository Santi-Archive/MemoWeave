import React from 'react';

interface FileIconProps {
  filename: string;
  filepath: string;
  selected: boolean;
  onClick: () => void;
}

export function FileIcon({ filename, filepath, selected, onClick }: FileIconProps) {
  const ext = filename.split('.').pop()?.toUpperCase() || 'TXT';

  return (
    <div className="flex flex-col items-center gap-1.5 cursor-pointer" onClick={onClick}>
      <div className={`
        w-[100px] h-[120px] rounded-lg flex items-center justify-center transition-all 
        ${selected 
          ? 'bg-gradient-to-b from-[#E8DFFF] to-[#D5C8F5] border-[3px] border-[#9B7EDC]' 
          : 'bg-gradient-to-b from-[#FAF8FE] to-[#F0EBFF] border-2 border-[#D7CFF1] hover:from-[#F1ECFF] hover:to-[#E5DCFF] hover:border-[#C4B3E8]'
        }
      `}>
        <div className="flex flex-col items-center gap-2">
          <div className="text-5xl">📄</div>
          <div className="bg-[#9B7EDC] text-white text-[10px] font-bold px-2 py-0.5 rounded uppercase">
            {ext}
          </div>
        </div>
      </div>
      <div className="max-w-[110px] text-center text-[11px] text-[#5A4A7A] break-words">
        {filename}
      </div>
    </div>
  );
}
