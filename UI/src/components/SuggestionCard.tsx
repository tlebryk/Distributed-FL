
import React from 'react';
import { Suggestion } from './CodeEditor';
import { cn } from '@/lib/utils';

interface SuggestionCardProps {
  suggestion: Suggestion;
  onClick: () => void;
  isActive?: boolean;
}

const SuggestionCard: React.FC<SuggestionCardProps> = ({
  suggestion,
  onClick,
  isActive = false,
}) => {
  return (
    <div 
      className={cn(
        "p-2 border rounded-md cursor-pointer transition-colors",
        isActive ? "bg-slate-100 border-slate-300" : "hover:bg-slate-50"
      )}
      onClick={onClick}
    >
      <div className="font-mono text-xs truncate">{suggestion.text}</div>
      {suggestion.description && (
        <div className="text-xs text-muted-foreground mt-1">{suggestion.description}</div>
      )}
    </div>
  );
};

export default SuggestionCard;
