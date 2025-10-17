const ChatMessage = ({ message, isUser, sources }) => {
  const groupedSources = {};

  if (sources && sources.length > 0) {
    sources.forEach((source) => {
      const fileName = source.split(':')[0];
      if (!groupedSources[fileName]) {
        groupedSources[fileName] = [];
      }
      groupedSources[fileName].push(source);
    });
  }

  return (
    <div className={`message ${isUser ? 'user' : 'bot'}`}>
      <div className="message-content">
        <div dangerouslySetInnerHTML={{ __html: marked.parse(message) }}></div>
        {!isUser && Object.keys(groupedSources).length > 0 && (
          <div className="sources-grouped">
            <strong>Sources:</strong>
            {Object.entries(groupedSources).map(([fileName, snippets], idx) => (
              <div key={idx} className="source-file">
                <div className="source-file-name">
                  <i className="fas fa-file-pdf"></i>
                  {fileName}
                </div>
                <div className="source-snippets">
                  {snippets.map((snippet, sidx) => (
                    <div key={sidx} className="source-snippet">
                      {snippet.split(':').slice(1).join(':')}
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
};
