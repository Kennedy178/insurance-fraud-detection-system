import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  ScanSearch,
  ClipboardList,
  ShieldCheck,
  Sun,
  Moon,
  Activity,
} from "lucide-react";
import { useTheme } from "../../context/ThemeContext";

interface NavItem {
  to: string;
  icon: React.ReactNode;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", icon: <LayoutDashboard size={18} />, label: "Dashboard" },
  { to: "/analyze", icon: <ScanSearch size={18} />, label: "Analyze Claim" },
  { to: "/history", icon: <ClipboardList size={18} />, label: "Claims History" },
];

interface SidebarProps {
  apiOnline: boolean;
}

export default function Sidebar({ apiOnline }: SidebarProps) {
  const { toggleTheme, isDark } = useTheme();
  const location = useLocation();

  return (
    <aside className="sidebar-bg flex flex-col w-60 min-h-screen fixed left-0 top-0 z-30 select-none">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b divider">
        <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-accent-muted flex-shrink-0">
          <ShieldCheck size={16} className="text-accent" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-display font-semibold text-primary leading-none">
            FraudGuard
          </p>
          <p className="text-xs font-mono text-secondary mt-0.5">v1.0.0</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="px-3 text-xs font-body font-semibold uppercase tracking-widest mb-3 text-muted">
          Main Menu
        </p>
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.to === "/"
              ? location.pathname === "/"
              : location.pathname.startsWith(item.to);

          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={`nav-item ${isActive ? "active" : ""}`}
            >
              <span className="flex-shrink-0">{item.icon}</span>
              <span>{item.label}</span>
              {isActive && (
                <span className="ml-auto w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-3 pb-4 space-y-2 border-t divider pt-4">
        {/* API status */}
        <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl card">
          <div className="relative flex-shrink-0">
            <Activity
              size={14}
              className={apiOnline ? "text-safe" : "text-fraud"}
            />
            <span
              className={`absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full animate-pulse-dot ${
                apiOnline ? "bg-safe" : "bg-fraud"
              }`}
            />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-body font-medium text-secondary leading-none">
              API Status
            </p>
            <p
              className={`text-xs font-mono font-semibold mt-0.5 ${
                apiOnline ? "text-safe" : "text-fraud"
              }`}
            >
              {apiOnline ? "Online" : "Offline"}
            </p>
          </div>
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="btn-ghost w-full justify-start text-xs"
          aria-label={`Switch to ${isDark ? "light" : "dark"} mode`}
        >
          {isDark ? (
            <Sun size={14} className="flex-shrink-0" />
          ) : (
            <Moon size={14} className="flex-shrink-0" />
          )}
          <span>{isDark ? "Light Mode" : "Dark Mode"}</span>
        </button>
      </div>
    </aside>
  );
}