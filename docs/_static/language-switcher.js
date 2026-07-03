(function () {
  "use strict";

  const languages = [
    { code: "en", label: "English", available: true },
    { code: "zh", label: "中文", available: true },
    { code: "ja", label: "日本語", available: true },
    { code: "ko", label: "한국어", available: true },
    { code: "de", label: "Deutsch", available: true },
    { code: "fr", label: "Français", available: true },
    { code: "ru", label: "Русский", available: true },
  ];

  const availableCodes = new Set(languages.filter((item) => item.available).map((item) => item.code));
  const languageLabels = new Map(languages.map((item) => [item.code, item.label]));

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

  function withFallbackNotice(path, requestedCode) {
    const url = new URL(path, window.location.origin);
    url.searchParams.set("pyna_fallback", requestedCode);
    return url.pathname + url.search + url.hash;
  }

  async function navigateWithFallback(requestedCode) {
    const requestedTarget = targetPath(requestedCode);
    const fallbackTarget = englishFallbackPath();

    if (requestedCode === "en" || requestedTarget === window.location.pathname) {
      window.location.assign(requestedTarget);
      return;
    }

    if (!availableCodes.has(requestedCode)) {
      window.location.assign(withFallbackNotice(fallbackTarget, requestedCode));
      return;
    }

    try {
      const response = await fetch(requestedTarget, { method: "HEAD", cache: "no-store" });
      window.location.assign(response.ok ? requestedTarget : withFallbackNotice(fallbackTarget, requestedCode));
    } catch (_error) {
      window.location.assign(withFallbackNotice(fallbackTarget, requestedCode));
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

  function buildFallbackBadge() {
    const params = new URLSearchParams(window.location.search);
    const fallbackCode = params.get("pyna_fallback");
    if (!fallbackCode || currentLanguage() !== "en" || document.querySelector(".pyna-translation-badge")) {
      return;
    }

    const article = document.querySelector(".bd-article, main.bd-content, main");
    if (!article) {
      return;
    }

    const cleanUrl = new URL(window.location.href);
    cleanUrl.searchParams.delete("pyna_fallback");
    const label = languageLabels.get(fallbackCode) || fallbackCode;

    const badge = document.createElement("div");
    badge.className = "pyna-translation-badge pyna-translation-badge--missing";
    badge.setAttribute("role", "note");

    const copy = document.createElement("div");
    copy.className = "pyna-translation-badge__copy";

    const title = document.createElement("span");
    title.className = "pyna-translation-badge__title";
    title.textContent = "Translation unavailable";

    const message = document.createElement("span");
    message.className = "pyna-translation-badge__message";
    message.textContent = `${label} is not available for this page yet; showing the English source.`;

    const link = document.createElement("a");
    link.className = "pyna-translation-badge__link";
    link.href = cleanUrl.pathname + cleanUrl.search + cleanUrl.hash;
    link.textContent = "English source";

    copy.append(title, message);
    badge.append(copy, link);
    article.prepend(badge);
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildSwitchers();
    hideOtherLanguageBranches();
    buildFallbackBadge();
  });
})();
