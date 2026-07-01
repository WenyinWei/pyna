(function () {
  "use strict";

  const languages = [
    { code: "en", label: "English", available: true },
    { code: "zh", label: "中文", available: true },
    { code: "ja", label: "日本語", available: false },
    { code: "de", label: "Deutsch", available: false },
    { code: "ko", label: "한국어", available: false },
    { code: "ru", label: "Русский", available: false },
  ];

  const availableCodes = new Set(languages.filter((item) => item.available).map((item) => item.code));

  function pathParts() {
    return window.location.pathname.split("/").filter(Boolean);
  }

  function docsRootIndex(parts) {
    const index = parts.findIndex((part) => availableCodes.has(part));
    if (index >= 0) {
      return index;
    }
    if (parts.length === 0) {
      return 0;
    }
    return parts[parts.length - 1].includes(".") ? parts.length - 1 : parts.length;
  }

  function currentLanguage() {
    const parts = pathParts();
    return parts.find((part) => availableCodes.has(part)) || "en";
  }

  function targetPath(requestedCode) {
    const code = availableCodes.has(requestedCode) ? requestedCode : "en";
    const parts = pathParts();
    const langIndex = parts.findIndex((part) => availableCodes.has(part));

    if (langIndex >= 0) {
      const next = parts.slice();
      next[langIndex] = code;
      return "/" + next.join("/");
    }

    const notebookIndex = parts.findIndex((part) => part === "notebooks");
    if (notebookIndex >= 0) {
      const root = parts.slice(0, notebookIndex);
      return "/" + root.concat([code, "tutorials", "index.html"]).join("/");
    }

    const rootIndex = docsRootIndex(parts);
    const root = parts.slice(0, rootIndex);
    return "/" + root.concat([code, "index.html"]).join("/");
  }

  function englishFallbackPath() {
    const parts = pathParts();
    const langIndex = parts.findIndex((part) => availableCodes.has(part));
    if (langIndex >= 0) {
      const next = parts.slice();
      next[langIndex] = "en";
      return "/" + next.join("/");
    }
    const notebookIndex = parts.findIndex((part) => part === "notebooks");
    if (notebookIndex >= 0) {
      const root = parts.slice(0, notebookIndex);
      return "/" + root.concat(["en", "tutorials", "index.html"]).join("/");
    }
    const rootIndex = docsRootIndex(parts);
    const root = parts.slice(0, rootIndex);
    return "/" + root.concat(["en", "index.html"]).join("/");
  }

  async function navigateWithFallback(requestedCode) {
    const requestedTarget = targetPath(requestedCode);
    const fallbackTarget = englishFallbackPath();

    if (requestedCode === "en" || requestedTarget === window.location.pathname) {
      window.location.assign(requestedTarget);
      return;
    }

    try {
      const response = await fetch(requestedTarget, { method: "HEAD", cache: "no-store" });
      window.location.assign(response.ok ? requestedTarget : fallbackTarget);
    } catch (_error) {
      window.location.assign(fallbackTarget);
    }
  }

  function createSwitcher() {
    const container = document.createElement("div");
    container.className = "pyna-language-switcher";
    container.dataset.pynaLanguageSwitcher = "true";

    const label = document.createElement("span");
    label.className = "pyna-language-switcher__label";
    label.textContent = "Language";

    const select = document.createElement("select");
    select.className = "pyna-language-select";
    select.setAttribute("aria-label", "Documentation language");
    container.appendChild(label);
    container.appendChild(select);
    return container;
  }

  function fillSelect(select) {
    if (select.options.length > 0) {
      return;
    }
    const active = currentLanguage();
    languages.forEach((language) => {
      const option = document.createElement("option");
      option.value = language.code;
      option.textContent = language.available ? language.label : `${language.label} -> English`;
      option.selected = language.code === active;
      select.appendChild(option);
    });
  }

  function wireSwitcher(container) {
    const select = container.querySelector("select");
    if (!select || select.dataset.pynaWired === "true") {
      return;
    }

    fillSelect(select);
    select.value = currentLanguage();
    select.addEventListener("change", (event) => {
      navigateWithFallback(event.target.value);
    });
    select.dataset.pynaWired = "true";
  }

  function buildSwitchers() {
    if (!document.querySelector(".pyna-language-switcher")) {
      const container = createSwitcher();
      const headerButtons = document.querySelector(
        ".navbar-header-items__end, .content-icon-container"
      );
      if (headerButtons) {
        headerButtons.prepend(container);
      } else {
        document.body.appendChild(container);
      }
    }

    document.querySelectorAll(".pyna-language-switcher").forEach(wireSwitcher);
  }

  function languageFromHref(href) {
    try {
      const url = new URL(href, window.location.href);
      return url.pathname.split("/").filter(Boolean).find((part) => availableCodes.has(part));
    } catch (_error) {
      return undefined;
    }
  }

  function hideOtherLanguageBranches() {
    const active = currentLanguage();
    document
      .querySelectorAll(".sidebar-tree a.reference, .toc-tree a.reference, .bd-sidebar-primary a, .bd-header a")
      .forEach((link) => {
        const linkLanguage = languageFromHref(link.getAttribute("href") || "");
        if (!linkLanguage || linkLanguage === active) {
          return;
        }
        const item = link.closest("li, .nav-item");
        if (item) {
          item.hidden = true;
        }
      });
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildSwitchers();
    hideOtherLanguageBranches();
  });
})();
