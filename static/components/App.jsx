const App = () => {
  const [messages, setMessages] = React.useState([]);
  const [message, setMessage] = React.useState('');
  const [isSending, setIsSending] = React.useState(false);
  const [config, setConfig] = React.useState(null);
  const [ingested, setIngested] = React.useState(null);
  const [stats, setStats] = React.useState(null);
  const [isDeleting, setIsDeleting] = React.useState(false);
  const [deleteResult, setDeleteResult] = React.useState(null);
  const messagesEndRef = React.useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadConfig = async () => {
    try {
      const response = await fetch('/config');
      const data = await response.json();
      setConfig(data);
    } catch (error) {
      console.error('Error loading config:', error);
    }
  };

  const loadIngested = async () => {
    try {
      const response = await fetch('/ingestion-status');
      const data = await response.json();
      setIngested(data.ingested || []);
    } catch (error) {
      console.error('Error loading ingested:', error);
      setIngested([]);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/embeddings/status');
      const data = await response.json();
      setStats(data.tables || {});
    } catch (error) {
      console.error('Error loading stats:', error);
      setStats({});
    }
  };

  React.useEffect(() => {
    loadConfig();
    loadIngested();
    loadStats();
    const interval = setInterval(() => {
      loadStats();
      loadIngested();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = message;
    setMessage('');
    setMessages((prev) => [...prev, { text: userMessage, isUser: true }]);
    setIsSending(true);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage }),
      });
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          text: data.answer || 'No response received',
          isUser: false,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: 'Sorry, there was an error processing your message.',
          isUser: false,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  const handleFileSelected = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    setMessages((prev) => [
      ...prev,
      { text: `Uploading ${file.name}...`, isUser: true },
    ]);

    try {
      const response = await fetch('/ingest', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            text: `✅ Successfully ingested ${file.name}`,
            isUser: false,
          },
        ]);
        loadIngested();
        loadStats();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            text: `❌ Error: ${data.detail || 'Failed to ingest file'}`,
            isUser: false,
          },
        ]);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessages((prev) => [
        ...prev,
        {
          text: `❌ Error uploading file: ${error.message}`,
          isUser: false,
        },
      ]);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete all embeddings? This cannot be undone.')) {
      return;
    }

    setIsDeleting(true);
    setDeleteResult(null);

    try {
      const response = await fetch('/embeddings/delete', {
        method: 'POST',
      });
      const data = await response.json();

      if (response.ok) {
        setDeleteResult({
          status: 'success',
          message: 'All embeddings deleted successfully',
        });
        loadStats();
        loadIngested();
      } else {
        setDeleteResult({
          status: 'error',
          message: data.detail || 'Failed to delete embeddings',
        });
      }
    } catch (error) {
      console.error('Error deleting embeddings:', error);
      setDeleteResult({
        status: 'error',
        message: `Error: ${error.message}`,
      });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="container">
      <div className="chat-section">
        <ChatHeader />
        <MessagesList messages={messages} messagesEndRef={messagesEndRef} />
        <ChatInput
          message={message}
          setMessage={setMessage}
          onSendMessage={handleSendMessage}
          onFileSelected={handleFileSelected}
          isSending={isSending}
        />
      </div>
      <Sidebar
        config={config}
        ingested={ingested}
        stats={stats}
        onDelete={handleDelete}
        isDeleting={isDeleting}
        deleteResult={deleteResult}
      />
    </div>
  );
};
