// ========================================
// MediCare AI - Professional Theme Toggle System
// ========================================

'use strict';

(function () {
    // Theme constants
    const THEME_KEY = 'medicare-theme';
    const DARK_THEME = 'dark';
    const LIGHT_THEME = 'light';

    // DOM Elements
    let themeToggle = null;

    // ========================================
    // Initialize Theme System
    // ========================================
    function initTheme() {
        // Get saved theme or detect system preference
        const savedTheme = localStorage.getItem(THEME_KEY);
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

        // Determine initial theme
        let initialTheme;
        if (savedTheme) {
            initialTheme = savedTheme;
        } else if (systemPrefersDark) {
            initialTheme = DARK_THEME;
        } else {
            initialTheme = LIGHT_THEME;
        }

        // Apply theme immediately (before DOM loads to prevent flash)
        applyTheme(initialTheme, false);

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupThemeToggle);
        } else {
            setupThemeToggle();
        }

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(THEME_KEY)) {
                applyTheme(e.matches ? DARK_THEME : LIGHT_THEME, true);
            }
        });
    }

    // ========================================
    // Setup Theme Toggle Button
    // ========================================
    function setupThemeToggle() {
        themeToggle = document.getElementById('themeToggle');

        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);

            // Add keyboard support
            themeToggle.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleTheme();
                }
            });
        }

        // Update toggle state based on current theme
        updateToggleState();
    }

    // ========================================
    // Toggle Theme
    // ========================================
    function toggleTheme(e) {
        const currentTheme = document.documentElement.getAttribute('data-theme') || LIGHT_THEME;
        const newTheme = currentTheme === DARK_THEME ? LIGHT_THEME : DARK_THEME;

        // Get click position for ripple effect
        let x = 50, y = 50;
        if (e && e.clientX) {
            x = (e.clientX / window.innerWidth) * 100;
            y = (e.clientY / window.innerHeight) * 100;
        }

        // Apply theme with animation
        applyTheme(newTheme, true, x, y);

        // Save preference
        localStorage.setItem(THEME_KEY, newTheme);

        // Show notification
        showThemeNotification(newTheme);
    }

    // ========================================
    // Apply Theme
    // ========================================
    function applyTheme(theme, animate = false, x = 50, y = 50) {
        if (animate) {
            // Create transition overlay for smooth animation
            const overlay = document.createElement('div');
            overlay.className = 'theme-transition-overlay';
            overlay.style.setProperty('--x', x + '%');
            overlay.style.setProperty('--y', y + '%');
            document.body.appendChild(overlay);

            // Trigger animation
            requestAnimationFrame(() => {
                overlay.classList.add('active');

                // Apply theme after short delay
                setTimeout(() => {
                    document.documentElement.setAttribute('data-theme', theme);
                    updateToggleState();

                    // Remove overlay
                    setTimeout(() => {
                        overlay.classList.remove('active');
                        setTimeout(() => overlay.remove(), 500);
                    }, 300);
                }, 150);
            });
        } else {
            // Apply immediately without animation
            document.documentElement.setAttribute('data-theme', theme);
            updateToggleState();
        }

        // Update meta theme-color for mobile browsers
        updateMetaThemeColor(theme);
    }

    // ========================================
    // Update Toggle Button State
    // ========================================
    function updateToggleState() {
        if (!themeToggle) return;

        const currentTheme = document.documentElement.getAttribute('data-theme') || LIGHT_THEME;
        const isDark = currentTheme === DARK_THEME;

        // Update aria-label for accessibility
        themeToggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
        themeToggle.setAttribute('title', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }

    // ========================================
    // Update Meta Theme Color
    // ========================================
    function updateMetaThemeColor(theme) {
        let metaThemeColor = document.querySelector('meta[name="theme-color"]');

        if (!metaThemeColor) {
            metaThemeColor = document.createElement('meta');
            metaThemeColor.name = 'theme-color';
            document.head.appendChild(metaThemeColor);
        }

        metaThemeColor.content = theme === DARK_THEME ? '#0d1117' : '#667eea';
    }

    // ========================================
    // Show Theme Change Notification
    // ========================================
    function showThemeNotification(theme) {
        // Check if notification function exists
        if (typeof showNotification === 'function') {
            const message = theme === DARK_THEME
                ? '🌙 Dark mode activated'
                : '☀️ Light mode activated';
            showNotification(message, 'info');
        }
    }

    // ========================================
    // Get Current Theme
    // ========================================
    function getCurrentTheme() {
        return document.documentElement.getAttribute('data-theme') || LIGHT_THEME;
    }

    // ========================================
    // Check if Dark Mode
    // ========================================
    function isDarkMode() {
        return getCurrentTheme() === DARK_THEME;
    }

    // ========================================
    // Export Functions
    // ========================================
    window.MediCareTheme = {
        toggle: toggleTheme,
        setTheme: applyTheme,
        getCurrentTheme: getCurrentTheme,
        isDarkMode: isDarkMode,
        DARK: DARK_THEME,
        LIGHT: LIGHT_THEME
    };

    // Initialize immediately
    initTheme();
})();

// ========================================
// Additional Theme-Related Utilities
// ========================================

// Smooth scroll with theme-aware colors
document.addEventListener('DOMContentLoaded', function () {
    // Add smooth transitions to all elements after page load
    setTimeout(() => {
        document.body.classList.add('theme-transitions-enabled');
    }, 100);
});

// Handle theme-aware charts and graphs
function updateChartColors() {
    const isDark = window.MediCareTheme && window.MediCareTheme.isDarkMode();

    // Update any Chart.js instances
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = isDark ? '#c9d1d9' : '#495057';
        Chart.defaults.borderColor = isDark ? '#30363d' : '#dee2e6';
    }
}

// Listen for theme changes to update charts
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.attributeName === 'data-theme') {
            updateChartColors();
        }
    });
});

observer.observe(document.documentElement, { attributes: true });
