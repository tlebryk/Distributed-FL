// CodeEditor.tsx
import React, { useState, useEffect } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { SuggestionService } from '@/services/suggestionService';
import SuggestionCard from './SuggestionCard';

export interface Suggestion {
  text: string;
  description?: string;
}

interface CodeEditorProps {
  initialCode?: string;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  initialCode = '# Write your Python code here\ndef hello_world():\n    print("Hello, World!")\n\nhello_world()',
}) => {
  const [code, setCode] = useState<string>(initialCode);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([
    { text: 'print("Hello, World!")', description: 'Print a greeting message' },
    { text: 'def function_name(param):', description: 'Define a new function' },
    { text: 'for item in items:', description: 'Loop through items' },
  ]);
  const [cursorPosition, setCursorPosition] = useState({ line: 0, ch: 0 });
  const [activeSuggestionIndex, setActiveSuggestionIndex] = useState<number>(-1);

  // Update suggestions whenever code changes with a slight delay
  useEffect(() => {
    // Using a debounce to avoid too many suggestion updates while typing
    const timer = setTimeout(() => {
      const newSuggestions = SuggestionService.getSuggestions(code, cursorPosition);
      setSuggestions(newSuggestions);
    }, 500); // 500ms delay

    return () => clearTimeout(timer);
  }, [code, cursorPosition]);

  const handleChange = React.useCallback((value: string) => {
    setCode(value);
    // get the cursor position from the editor here
    // and use it for more accurate suggestions
  }, []);

  const insertSuggestion = (suggestion: Suggestion) => {
    // TODO: edit to insert text at cursor position
    setCode(code + '\n' + suggestion.text);
    setActiveSuggestionIndex(-1);
  };

  // Handle keyboard navigation for suggestions
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestionIndex(prev =>
        prev < suggestions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestionIndex(prev =>
        prev > 0 ? prev - 1 : suggestions.length - 1
      );
    } else if (e.key === 'Enter' && activeSuggestionIndex >= 0) {
      e.preventDefault();
      insertSuggestion(suggestions[activeSuggestionIndex]);
    } else if (e.key === 'Escape') {
      setActiveSuggestionIndex(-1);
    }
  };

  return (
    <div className="flex flex-col md:flex-row gap-4 w-full h-full">
      <Card className="flex-1">
        <CardHeader>
          <CardTitle>Python Code Editor</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="border rounded-md overflow-hidden">
            <div
              onKeyDown={handleKeyDown}
              tabIndex={0}
              className="focus:outline-none"
            >
              <CodeMirror
                value={code}
                height="400px"
                extensions={[python()]}
                onChange={handleChange}
                theme="dark"
                className="text-sm"
                // for now use basic tracking without detailed cursor information
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLine: true
                }}
                onUpdate={(update) => {
                  // This is just a placeholder for true cursor tracking
                  if (update.docChanged) {
                    // When document changes, just use a basic approximation
                    // Implement more accurate cursor tracking here
                    setCursorPosition({
                      line: 0,
                      ch: 0
                    });
                  }
                }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="w-full md:w-64">
        <CardHeader>
          <CardTitle>Suggestions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {suggestions.map((suggestion, index) => (
              <SuggestionCard
                key={index}
                suggestion={suggestion}
                onClick={() => insertSuggestion(suggestion)}
                isActive={index === activeSuggestionIndex}
              />
            ))}
            <Separator className="my-2" />
            <div className="p-2 bg-blue-50 border border-blue-100 rounded-md text-xs text-blue-500 italic">
              <div className="font-semibold">AI Integration Placeholder</div>
              <div className="mt-1">Static examples above, will edit these suggestins to use our federated learning model</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CodeEditor;
