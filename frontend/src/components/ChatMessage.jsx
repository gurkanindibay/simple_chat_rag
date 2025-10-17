import { marked } from 'marked';

export function ChatMessage({ message, isUser, sources }) {
  const groupedSources = {};

  if (sources && Array.isArray(sources) && sources.length > 0) {
    sources.forEach((source) => {
      // Handle both string and object sources
      const fileName = typeof source === 'string' 
        ? source.split(':')[0] 
        : source.source || 'Unknown';
      
      if (!groupedSources[fileName]) {
        groupedSources[fileName] = [];
      }
      
      const sourceText = typeof source === 'string' 
        ? source 
        : `${source.source}: ${source.text}`;
      
      groupedSources[fileName].push(sourceText);
    });
  }

  return (
    <div className={`message ${isUser ? 'user' : 'bot'}`}>
      <div className="message-content">
        {isUser ? (
          <p>{message}</p>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: marked.parse(message) }} />
        )}
        
        {!isUser && Object.keys(groupedSources).length > 0 && (
          <div className="sources-grouped">
            <strong>📚 Sources:</strong>
            {Object.entries(groupedSources).map(([fileName, snippets]) => (
              <div key={fileName} className="source-file">
                <div className="source-file-name">
                  <i className="fas fa-file-pdf"></i> {fileName}
                </div>
                <div className="source-snippets">
                  {snippets.map((snippet, idx) => (
                    <div key={idx} className="source-snippet">
                      {snippet}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
