(function () {
    "use strict";

    const COLLAPSE_LINE_THRESHOLD = 72;
    const PREVIEW_LINES = 12;

    const labels = {
        en: {
            show: "Show full code",
            hide: "Collapse code",
            preview: "Code preview",
            lines: "lines",
        },
        zh: {
            show: "展开完整代码",
            hide: "收起代码",
            preview: "代码预览",
            lines: "行",
        },
        ja: {
            show: "コードを展開",
            hide: "コードを折りたたむ",
            preview: "コードプレビュー",
            lines: "行",
        },
        ko: {
            show: "전체 코드 보기",
            hide: "코드 접기",
            preview: "코드 미리보기",
            lines: "줄",
        },
        de: {
            show: "Code anzeigen",
            hide: "Code einklappen",
            preview: "Codevorschau",
            lines: "Zeilen",
        },
        fr: {
            show: "Afficher le code",
            hide: "Replier le code",
            preview: "Apercu du code",
            lines: "lignes",
        },
        ru: {
            show: "Показать код",
            hide: "Свернуть код",
            preview: "Предпросмотр кода",
            lines: "строк",
        },
    };
    const languageCodes = new Set(Object.keys(labels));

    function pageLanguage() {
        const configured = document.documentElement.dataset.pynaLang;
        if (languageCodes.has(configured)) {
            return configured;
        }

        const parts = window.location.pathname.split("/").filter(Boolean);
        const notebookIndex = parts.findIndex((part) => part === "notebooks");
        if (
            notebookIndex >= 0 &&
            parts[notebookIndex + 1] === "i18n" &&
            languageCodes.has(parts[notebookIndex + 2])
        ) {
            document.documentElement.dataset.pynaLang = parts[notebookIndex + 2];
            return parts[notebookIndex + 2];
        }

        const pathLang = parts.find((part) => languageCodes.has(part)) || "en";
        document.documentElement.dataset.pynaLang = pathLang;
        return pathLang;
    }

    function text() {
        return labels[pageLanguage()] || labels.en;
    }

    function codeLineCount(pre) {
        const raw = pre.textContent || "";
        if (!raw.trim()) {
            return 0;
        }
        return raw.replace(/\s+$/u, "").split(/\r?\n/u).length;
    }

    function hasAnyClass(element, names) {
        return names.some((name) => element.classList.contains(name));
    }

    function requestedMode(cell) {
        if (hasAnyClass(cell, ["pyna-keep-input", "pyna-show-input", "keep-input"])) {
            return "keep";
        }
        if (hasAnyClass(cell, ["pyna-hide-input", "hide-input"])) {
            return "hide";
        }
        if (hasAnyClass(cell, ["pyna-collapse-input", "pyna-fold-input", "collapse-input"])) {
            return "collapse";
        }
        return null;
    }

    function buttonLabel(expanded, lineCount) {
        const copy = text();
        if (expanded) {
            return copy.hide;
        }
        return `${copy.show} (${lineCount} ${copy.lines})`;
    }

    function setExpanded(cell, button, expanded) {
        const totalLines = Number.parseInt(cell.dataset.pynaInputLines || "0", 10);
        cell.classList.toggle("pyna-nbinput-expanded", expanded);
        cell.classList.toggle("pyna-nbinput-collapsed", !expanded);
        button.setAttribute("aria-expanded", expanded ? "true" : "false");
        button.setAttribute("aria-label", buttonLabel(expanded, totalLines));
        button.textContent = buttonLabel(expanded, totalLines);
    }

    function makeToggleBar(cell, lineCount, hiddenPreview) {
        const bar = document.createElement("div");
        bar.className = "pyna-code-togglebar";

        const summary = document.createElement("span");
        summary.className = "pyna-code-summary";
        summary.textContent = `${text().preview}: ${lineCount} ${text().lines}`;
        bar.appendChild(summary);

        const button = document.createElement("button");
        button.type = "button";
        button.className = "pyna-code-toggle";
        button.setAttribute("aria-expanded", "false");
        button.setAttribute("aria-label", buttonLabel(false, lineCount));
        button.textContent = buttonLabel(false, lineCount);
        button.addEventListener("click", function () {
            setExpanded(cell, button, !cell.classList.contains("pyna-nbinput-expanded"));
        });
        bar.appendChild(button);

        if (hiddenPreview) {
            bar.classList.add("pyna-code-togglebar--hidden-preview");
        }
        return { bar, button };
    }

    function prepareNotebookInput(cell) {
        if (cell.dataset.pynaInputReady === "true") {
            return;
        }

        const inputArea = cell.querySelector(".input_area");
        const pre = inputArea ? inputArea.querySelector("pre") : null;
        if (!inputArea || !pre) {
            return;
        }

        const mode = requestedMode(cell);
        const lineCount = codeLineCount(pre);
        if (mode === "keep" || (mode === null && lineCount < COLLAPSE_LINE_THRESHOLD)) {
            return;
        }

        const hiddenPreview = mode === "hide";
        const previewLines = hiddenPreview ? 0 : PREVIEW_LINES;
        cell.dataset.pynaInputReady = "true";
        cell.dataset.pynaInputLines = String(lineCount);
        cell.style.setProperty("--pyna-code-preview-lines", String(previewLines));
        cell.classList.add("pyna-nbinput-foldable", "pyna-nbinput-collapsed");
        if (hiddenPreview) {
            cell.classList.add("pyna-nbinput-hidden-preview");
        }

        const toggle = makeToggleBar(cell, lineCount, hiddenPreview);
        inputArea.parentNode.insertBefore(toggle.bar, inputArea);
        setExpanded(cell, toggle.button, false);
    }

    function initNotebookInputs() {
        document.querySelectorAll(".nbinput").forEach(prepareNotebookInput);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initNotebookInputs);
    } else {
        initNotebookInputs();
    }
}());
