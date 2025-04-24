
import { useState } from 'react';
import CodeEditor from '@/components/CodeEditor';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const Index = () => {
  const [output, setOutput] = useState<string>('');

  const runCode = () => {
    // Edit to execute the code or send it to a backend
    setOutput("Code execution is not implemented yet.\nThis is where the output would appear after running your Python code.");
  };

  return (
    <div className="container mx-auto p-4 max-w-7xl">
      <h1 className="text-3xl font-bold mb-4">Theo and Richael's Project!!</h1>
      <p className="text-lg text-muted-foreground mb-2">
        A simple Python editor with AI-powered code suggestions
      </p>      
      <div className="grid grid-cols-1 gap-6">
        <CodeEditor />
        
        <div className="flex justify-end">
          <Button onClick={runCode}>Run Code</Button>
        </div>
        
        <Card className="p-4">
          <h2 className="text-xl font-semibold mb-2">Output</h2>
          <div className="bg-slate-950 text-slate-50 p-4 rounded-md font-mono text-sm whitespace-pre-wrap h-[150px] overflow-y-auto">
            {output || 'Run your code to see output here'}
          </div>
        </Card>

        <Tabs defaultValue="about">
          <TabsList>
            <TabsTrigger value="about">About</TabsTrigger>
            <TabsTrigger value="usage">How to use</TabsTrigger>
          </TabsList>
          <TabsContent value="about" className="p-4">
            <h3 className="font-medium mb-2">About Our Project</h3>
            <p>
              This is a simple Python code editor with placeholders for AI-powered code suggestions using federated learning
            </p>
          </TabsContent>
          <TabsContent value="usage" className="p-4">
            <h3 className="font-medium mb-2">How to use:</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li>Type your Python code in the editor</li>
              <li>See suggestions appear in the right panel</li>
              <li>Click on a suggestion to insert it into your code</li>
              <li>Press "Run Code" to execute (not yet implemented)</li>
            </ul>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
