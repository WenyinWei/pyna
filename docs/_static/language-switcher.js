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
  const chromeTranslations = {
    en: {
      language: "Language",
      documentationLanguage: "Documentation language",
      search: "Search",
      searchDocs: "Search the docs ...",
      skipMain: "Skip to main content",
      backToTop: "Back to top",
      siteNavigation: "Site navigation",
      sectionNavigation: "Contents",
      onThisPage: "On this page",
      collapseSidebar: "Collapse Sidebar",
      expandSidebar: "Expand Sidebar",
      colorMode: "Color mode",
      systemSettings: "System Settings",
      light: "Light",
      dark: "Dark",
    },
    zh: {
      language: "语言",
      documentationLanguage: "文档语言",
      search: "搜索",
      searchDocs: "搜索文档 ...",
      skipMain: "跳到正文",
      backToTop: "返回顶部",
      siteNavigation: "站点导航",
      sectionNavigation: "目录",
      onThisPage: "本页目录",
      collapseSidebar: "收起侧栏",
      expandSidebar: "展开侧栏",
      colorMode: "颜色模式",
      systemSettings: "跟随系统",
      light: "浅色",
      dark: "深色",
    },
    ja: {
      language: "言語",
      documentationLanguage: "ドキュメント言語",
      search: "検索",
      searchDocs: "ドキュメントを検索 ...",
      skipMain: "本文へ移動",
      backToTop: "先頭へ戻る",
      siteNavigation: "サイトナビゲーション",
      sectionNavigation: "目次",
      onThisPage: "このページ",
      collapseSidebar: "サイドバーを折りたたむ",
      expandSidebar: "サイドバーを展開",
      colorMode: "配色モード",
      systemSettings: "システム設定",
      light: "ライト",
      dark: "ダーク",
    },
    ko: {
      language: "언어",
      documentationLanguage: "문서 언어",
      search: "검색",
      searchDocs: "문서 검색 ...",
      skipMain: "본문으로 이동",
      backToTop: "맨 위로",
      siteNavigation: "사이트 탐색",
      sectionNavigation: "목차",
      onThisPage: "이 페이지",
      collapseSidebar: "사이드바 접기",
      expandSidebar: "사이드바 펼치기",
      colorMode: "색상 모드",
      systemSettings: "시스템 설정",
      light: "라이트",
      dark: "다크",
    },
    de: {
      language: "Sprache",
      documentationLanguage: "Dokumentationssprache",
      search: "Suchen",
      searchDocs: "Dokumentation durchsuchen ...",
      skipMain: "Zum Hauptinhalt springen",
      backToTop: "Nach oben",
      siteNavigation: "Seitennavigation",
      sectionNavigation: "Inhalt",
      onThisPage: "Auf dieser Seite",
      collapseSidebar: "Seitenleiste einklappen",
      expandSidebar: "Seitenleiste ausklappen",
      colorMode: "Farbmodus",
      systemSettings: "Systemeinstellung",
      light: "Hell",
      dark: "Dunkel",
    },
    fr: {
      language: "Langue",
      documentationLanguage: "Langue de la documentation",
      search: "Rechercher",
      searchDocs: "Rechercher dans la documentation ...",
      skipMain: "Aller au contenu principal",
      backToTop: "Retour en haut",
      siteNavigation: "Navigation du site",
      sectionNavigation: "Sommaire",
      onThisPage: "Sur cette page",
      collapseSidebar: "Replier la barre latérale",
      expandSidebar: "Déplier la barre latérale",
      colorMode: "Mode de couleur",
      systemSettings: "Paramètres système",
      light: "Clair",
      dark: "Sombre",
    },
    ru: {
      language: "Язык",
      documentationLanguage: "Язык документации",
      search: "Поиск",
      searchDocs: "Поиск по документации ...",
      skipMain: "Перейти к основному содержанию",
      backToTop: "Наверх",
      siteNavigation: "Навигация по сайту",
      sectionNavigation: "Содержание",
      onThisPage: "На этой странице",
      collapseSidebar: "Свернуть боковую панель",
      expandSidebar: "Развернуть боковую панель",
      colorMode: "Цветовой режим",
      systemSettings: "Системные настройки",
      light: "Светлая",
      dark: "Темная",
    },
  };

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

  function configuredLanguage() {
    const lang = document.documentElement.dataset.pynaLang;
    return availableCodes.has(lang) ? lang : undefined;
  }

  function notebookPath(parts) {
    const index = parts.findIndex((part) => part === "notebooks");
    if (index < 0) {
      return null;
    }

    const tail = parts.slice(index + 1);
    if (tail[0] === "i18n" && availableCodes.has(tail[1])) {
      return {
        root: parts.slice(0, index),
        language: tail[1],
        suffix: tail.slice(2),
      };
    }

    return {
      root: parts.slice(0, index),
      language: undefined,
      suffix: tail,
    };
  }

  function currentLanguage() {
    const configured = configuredLanguage();
    if (configured) {
      return configured;
    }

    const parts = pathParts();
    const notebook = notebookPath(parts);
    return (notebook && notebook.language) || parts.find((part) => availableCodes.has(part)) || "en";
  }

  function targetPath(requestedCode) {
    const code = availableCodes.has(requestedCode) ? requestedCode : "en";
    const parts = pathParts();
    const notebook = notebookPath(parts);
    if (notebook) {
      const suffix = notebook.suffix.length > 0 ? notebook.suffix : ["tutorials", "index.html"];
      return "/" + notebook.root.concat(["notebooks", "i18n", code], suffix).join("/");
    }

    const langIndex = parts.findIndex((part) => availableCodes.has(part));

    if (langIndex >= 0) {
      const next = parts.slice();
      next[langIndex] = code;
      return "/" + next.join("/");
    }

    const rootIndex = docsRootIndex(parts);
    const root = parts.slice(0, rootIndex);
    return "/" + root.concat([code, "index.html"]).join("/");
  }

  function englishFallbackPath() {
    return targetPath("en");
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
    label.textContent = chromeTranslations[currentLanguage()].language;

    const select = document.createElement("select");
    select.className = "pyna-language-select";
    select.setAttribute("aria-label", chromeTranslations[currentLanguage()].documentationLanguage);
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
      if (url.origin !== window.location.origin) {
        return undefined;
      }

      const parts = url.pathname.split("/").filter(Boolean);
      const notebook = notebookPath(parts);
      return (notebook && notebook.language) || parts.find((part) => availableCodes.has(part));
    } catch (_error) {
      return undefined;
    }
  }

  function containsCurrentPage(element) {
    return Boolean(
      element &&
        (element.matches(".current, .active, [aria-current='page']") ||
          element.querySelector(".current, .active, [aria-current='page']"))
    );
  }

  function directDetails(element) {
    return Array.from(element.children).find((child) => child.tagName === "DETAILS");
  }

  function keepCurrentBranchVisible() {
    document
      .querySelectorAll(".bd-sidebar-primary .current, .sidebar-tree .current, .toc-tree .current, a[aria-current='page']")
      .forEach((current) => {
        let item = current.closest("li, .nav-item");
        while (item) {
          item.hidden = false;
          const details = directDetails(item);
          if (details) {
            details.open = true;
          }
          item = item.parentElement ? item.parentElement.closest("li, .nav-item") : null;
        }
      });
  }

  function hideOtherLanguageBranches() {
    const active = currentLanguage();
    document
      .querySelectorAll(
        ".sidebar-tree a.reference, .toc-tree a.reference, " +
          ".bd-header .bd-navbar-elements a, " +
          ".bd-sidebar-primary .sidebar-header-items a"
      )
      .forEach((link) => {
        const linkLanguage = languageFromHref(link.getAttribute("href") || "");
        if (!linkLanguage || linkLanguage === active) {
          return;
        }
        const item = link.closest("li, .nav-item");
        if (item) {
          if (containsCurrentPage(item)) {
            item.hidden = false;
            return;
          }
          item.hidden = true;
        }
      });
    keepCurrentBranchVisible();
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

  function replaceTextAfterIcon(element, text) {
    if (!element) {
      return;
    }
    Array.from(element.childNodes).forEach((node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        node.remove();
      }
    });
    element.append(document.createTextNode(text));
  }

  function replaceButtonText(element, text) {
    if (!element) {
      return;
    }
    Array.from(element.childNodes).forEach((node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        node.remove();
      }
    });
    element.append(document.createTextNode(text));
  }

  function localizeChrome() {
    const lang = currentLanguage();
    const text = chromeTranslations[lang] || chromeTranslations.en;
    document.documentElement.lang = lang;

    document.querySelectorAll(".pyna-language-switcher__label").forEach((label) => {
      label.textContent = text.language;
    });
    document.querySelectorAll(".pyna-language-select").forEach((select) => {
      select.setAttribute("aria-label", text.documentationLanguage);
    });

    document.querySelectorAll(".search-button__button").forEach((button) => {
      button.setAttribute("title", text.search);
      button.setAttribute("aria-label", text.search);
    });
    document.querySelectorAll("#pst-search-dialog input[type='search']").forEach((input) => {
      input.setAttribute("placeholder", text.searchDocs);
      input.setAttribute("aria-label", text.searchDocs);
    });

    const skipLink = document.querySelector("#pst-skip-link a");
    if (skipLink) {
      skipLink.textContent = text.skipMain;
    }
    replaceTextAfterIcon(document.querySelector("#pst-back-to-top"), text.backToTop);

    document.querySelectorAll(".primary-toggle").forEach((button) => {
      button.setAttribute("aria-label", text.siteNavigation);
    });
    document.querySelectorAll(".secondary-toggle").forEach((button) => {
      button.setAttribute("aria-label", text.onThisPage);
    });
    document.querySelectorAll(".bd-docs-nav").forEach((nav) => {
      nav.setAttribute("aria-label", text.sectionNavigation);
    });
    document.querySelectorAll(".bd-links__title").forEach((heading) => {
      heading.textContent = text.sectionNavigation;
    });
    document.querySelectorAll(".onthispage").forEach((heading) => {
      replaceTextAfterIcon(heading, text.onThisPage);
    });
    document.querySelectorAll(".pst-collapse-sidebar-label").forEach((label) => {
      label.textContent = text.collapseSidebar;
    });
    document.querySelectorAll(".pst-expand-sidebar-label").forEach((label) => {
      label.textContent = text.expandSidebar;
    });

    document.querySelectorAll(".theme-switch-container").forEach((container) => {
      container.setAttribute("title", text.colorMode);
    });
    document.querySelectorAll(".theme-switch-button").forEach((button) => {
      button.setAttribute("aria-label", text.colorMode);
    });
    document.querySelectorAll(".theme-switch[data-mode='auto']").forEach((icon) => {
      icon.setAttribute("title", text.systemSettings);
    });
    document.querySelectorAll(".theme-switch[data-mode='light']").forEach((icon) => {
      icon.setAttribute("title", text.light);
    });
    document.querySelectorAll(".theme-switch[data-mode='dark']").forEach((icon) => {
      icon.setAttribute("title", text.dark);
    });
    document.querySelectorAll(".theme-change-button[data-mode='auto']").forEach((button) => {
      replaceButtonText(button, text.systemSettings);
    });
    document.querySelectorAll(".theme-change-button[data-mode='light']").forEach((button) => {
      replaceButtonText(button, text.light);
    });
    document.querySelectorAll(".theme-change-button[data-mode='dark']").forEach((button) => {
      replaceButtonText(button, text.dark);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    buildSwitchers();
    hideOtherLanguageBranches();
    buildFallbackBadge();
    localizeChrome();
  });
})();
