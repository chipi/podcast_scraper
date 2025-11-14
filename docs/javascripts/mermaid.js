document$.subscribe(function() {
  mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {
      useMaxWidth: true,
      htmlLabels: true,
      curve: 'basis'
    }
  });
  
  // Process all mermaid diagrams on the page
  mermaid.run();
});

