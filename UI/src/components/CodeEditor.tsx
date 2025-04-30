// src/components/CodeEditor.tsx
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
  initialCode = `# Write your Python code here
def hello_world():
    print("Hello, World!")

hello_world()`,
}) => {
  /* ----------------------------- state ----------------------------- */
  const [code, setCode] = useState<string>(initialCode);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([
    { text: 'print("Hello, World!")', description: 'Print a greeting message' },
    { text: 'def function_name(param):', description: 'Define a new function' },
    { text: 'for item in items:', description: 'Loop through items' },
  ]);
  const [cursorPosition, setCursorPosition] = useState({ line: 0, ch: 0 });
  const [activeSuggestionIndex, setActiveSuggestionIndex] =
    useState<number>(-1);

  // NEW: store the generated text returned from the inference endpoint
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);

  /* ------------------------ suggestions updater -------------------- */
  useEffect(() => {
    const timer = setTimeout(() => {
      const newSuggestions = SuggestionService.getSuggestions(
        code,
        cursorPosition
      );
      setSuggestions(newSuggestions);
    }, 500);

    return () => clearTimeout(timer);
  }, [code, cursorPosition]);

  /* ---------------------------- helpers ---------------------------- */
  const handleChange = React.useCallback((value: string) => {
    setCode(value);
    // TODO: supply real cursor info if you need context-aware suggestions
  }, []);

  const insertSuggestion = (suggestion: Suggestion) => {
    // naïve insertion (appends). Replace with cursor logic later
    setCode((prev) => `${prev}\n${suggestion.text}`);
    setActiveSuggestionIndex(-1);
  };

  // generate button handler
  const handleGenerate = async () => {
    setGeneratedCode(null); // clear previous
    try {
      const { generated_text } = await SuggestionService.generateCode(code);
      setGeneratedCode(generated_text);
    } catch (err) {
      console.error(err);
      setGeneratedCode(`Error: ${(err as Error).message}`);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveSuggestionIndex((prev) =>
        prev < suggestions.length - 1 ? prev + 1 : 0
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveSuggestionIndex((prev) =>
        prev > 0 ? prev - 1 : suggestions.length - 1
      );
    } else if (e.key === 'Enter' && activeSuggestionIndex >= 0) {
      e.preventDefault();
      insertSuggestion(suggestions[activeSuggestionIndex]);
    } else if (e.key === 'Escape') {
      setActiveSuggestionIndex(-1);
    }
  };

  /* ---------------------------- render ----------------------------- */
  return (
    <div className="flex flex-col md:flex-row gap-4 w-full h-full">
      {/* ---------------------- editor card ------------------------- */}
      <Card className="flex-1">
        <CardHeader>
          <CardTitle>Python Code Editor</CardTitle>
        </CardHeader>

        <CardContent>
          {/* code mirror */}
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
                basicSetup={{ lineNumbers: true, highlightActiveLine: true }}
                onUpdate={(update) => {
                  if (update.docChanged) {
                    setCursorPosition({ line: 0, ch: 0 });
                  }
                }}
              />
            </div>
          </div>

          {/* generate button & output */}
          <div className="mt-4 space-y-4">
            <button
              onClick={handleGenerate}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Generate
            </button>

            {generatedCode && (
              <div className="p-2 border rounded bg-gray-800 text-white text-sm whitespace-pre-wrap">
                {generatedCode}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* ------------------- suggestions sidebar -------------------- */}
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
              <div className="mt-1">
                Static examples above; these will be replaced by our federated
                model.
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CodeEditor;
