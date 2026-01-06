// Display deployment metadata in the footer
(function() {
  function loadDeployInfo() {
    // Load deployment info JSON (absolute path from site root)
    // Site URL is https://chipi.github.io/podcast_scraper/ so base path is /podcast_scraper/
    fetch('/podcast_scraper/deploy-info.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Deploy info not found');
        }
        return response.json();
      })
      .then(data => {
        // Format timestamp for display
        const timestamp = new Date(data.timestamp.replace(' UTC', 'Z'));
        const formattedDate = timestamp.toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          timeZone: 'UTC',
          timeZoneName: 'short'
        });

        // Create deployment info element
        const deployInfo = document.createElement('div');
        deployInfo.className = 'md-footer-deploy-info';
        deployInfo.innerHTML = `
          <span class="deploy-label">📄 Docs updated:</span>
          <span class="deploy-date">${formattedDate}</span>
          <span class="deploy-separator">•</span>
          <span class="deploy-branch">${data.branch}</span>
          <span class="deploy-separator">@</span>
          <a href="${data.commitUrl}" class="deploy-commit" target="_blank" rel="noopener">${data.commit}</a>
        `;

        // Find the footer and insert deployment info
        const footer = document.querySelector('.md-footer');
        if (footer) {
          // Check if already added
          if (footer.querySelector('.md-footer-deploy-info')) {
            return;
          }
          // Insert before the footer inner content
          const footerInner = footer.querySelector('.md-footer__inner');
          if (footerInner) {
            footerInner.insertBefore(deployInfo, footerInner.firstChild);
          } else {
            footer.insertBefore(deployInfo, footer.firstChild);
          }
        }
      })
      .catch(error => {
        // Silently fail if deploy-info.json doesn't exist (e.g., local builds)
        console.debug('Deployment info not available:', error);
      });
  }

  // Try to load immediately if DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadDeployInfo);
  } else {
    loadDeployInfo();
  }

  // Also subscribe to Material theme navigation events if available
  if (typeof document$ !== 'undefined') {
    document$.subscribe(loadDeployInfo);
  }
})();

