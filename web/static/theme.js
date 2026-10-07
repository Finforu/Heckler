// Applies the saved theme before the page paints (loaded in <head>, not a module).
try {
  var theme = JSON.parse(localStorage.getItem('heckler.theme') || '"system"');
  if (theme === 'light' || theme === 'dark') document.documentElement.setAttribute('data-theme', theme);
} catch (e) { /* storage blocked: follow the system */ }
