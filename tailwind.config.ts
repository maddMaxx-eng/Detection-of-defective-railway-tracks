import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#070d19", 900: "#0b1324", 800: "#101b30", 700: "#16233c", 600: "#1f3050", 500: "#2c4166" },
        steel: { 100: "#e6ebf3", 200: "#c3cddd", 300: "#9aa8bf", 400: "#7686a2" },
        signal: { DEFAULT: "#ff7a1a", dim: "#c95d0e", soft: "rgba(255,122,26,0.12)" },
      },
      fontFamily: {
        display: ['Bahnschrift', '"DIN Alternate"', '"Roboto Condensed"', '"Arial Narrow"', "system-ui", "sans-serif"],
        sans: ["system-ui", "-apple-system", '"Segoe UI"', "Roboto", "Helvetica", "Arial", "sans-serif"],
      },
      borderRadius: { card: "14px" },
      keyframes: {
        scan: { "0%": { transform: "translateY(-10%)" }, "100%": { transform: "translateY(110%)" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { scan: "scan 3.6s linear infinite", shimmer: "shimmer 1.6s infinite" },
    },
  },
  plugins: [],
};
export default config;
