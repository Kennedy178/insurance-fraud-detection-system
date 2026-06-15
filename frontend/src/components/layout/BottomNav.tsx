import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  ScanSearch,
  ClipboardList,
  Sun,
  Moon,
} from "lucide-react";
import { useTheme } from "../../context/ThemeContext";

interface BottomNavProps {
  apiOnline: boolean;
}

const NAV_ITEMS = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/analyze", icon: ScanSearch, label: "Analyze" },
  { to: "/history", icon: ClipboardList, label: "History" },
];

export default function BottomNav({ apiOnline }: BottomNavProps) {
  const { toggleTheme, isDark } = useTheme();
  const location = useLocation();

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 flex items-center justify-around px-2 py-2 safe-area-bottom"
      style={{
        background: isDark ? "rgba(10,14,26,0.95)" : "rgba(255,255,255,0.95)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderTop: isDark
          ? "1px solid rgba(255,255,255,0.07)"
          : "1px solid rgba(0,0,0,0.08)",
      }}
    >
      {NAV_ITEMS.map((item) => {
        const Icon = item.icon;
        const isActive =
          item.to === "/"
            ? location.pathname === "/"
            : location.pathname.startsWith(item.to);

        return (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className="flex flex-col items-center gap-1 px-4 py-1.5 rounded-xl transition-all duration-200 flex-1"
            style={{
              color: isActive
                ? "#4f7ef7"
                : isDark
                ? "#7c8fa6"
                : "#4a5568",
              background: isActive
                ? "rgba(79,126,247,0.1)"
                : "transparent",
            }}
          >
            <Icon size={20} />
            <span className="text-2xs font-body font-medium">{item.label}</span>
          </NavLink>
        );
      })}

      {/* Theme toggle as 4th tab */}
      <button
        onClick={toggleTheme}
        className="flex flex-col items-center gap-1 px-4 py-1.5 rounded-xl transition-all duration-200 flex-1"
        style={{ color: isDark ? "#7c8fa6" : "#4a5568" }}
        aria-label="Toggle theme"
      >
        {isDark ? <Sun size={20} /> : <Moon size={20} />}
        <span className="text-2xs font-body font-medium">
          {isDark ? "Light" : "Dark"}
        </span>
      </button>

      {/* API status dot - far right corner indicator */}
      <span
        className="absolute top-2 right-3 w-1.5 h-1.5 rounded-full"
        style={{ background: apiOnline ? "#22c55e" : "#f04f57" }}
      />
    </nav>
  );
}