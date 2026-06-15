import { Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";
import BottomNav from "./BottomNav";
import { getHealth } from "../../api/client";

export default function Layout() {
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    const check = async () => {
      try {
        const health = await getHealth();
        setApiOnline(health.status === "healthy" || health.model_loaded);
      } catch {
        setApiOnline(false);
      }
    };

    check();
    const interval = setInterval(check, 30_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex min-h-screen">
      {/* Sidebar - hidden on mobile, visible from md up */}
      <div className="hidden md:flex">
        <Sidebar apiOnline={apiOnline} />
      </div>

      {/* Main area - full width on mobile, offset by sidebar on md+ */}
      <div className="flex-1 flex flex-col min-h-screen md:ml-60">
        <Topbar apiOnline={apiOnline} />

        {/* Page content - extra bottom padding on mobile for bottom nav */}
        <main className="flex-1 p-4 md:p-6 pb-24 md:pb-6">
          <Outlet />
        </main>
      </div>

      {/* Bottom nav - visible on mobile only */}
      <div className="md:hidden">
        <BottomNav apiOnline={apiOnline} />
      </div>
    </div>
  );
}