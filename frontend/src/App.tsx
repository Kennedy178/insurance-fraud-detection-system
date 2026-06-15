import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./context/ThemeContext";
import Layout from "./components/layout/Layout";
import Dashboard from "./pages/Dashboard";
import AnalyzeClaim from "./pages/AnalyzeClaim";
import ClaimsHistory from "./pages/ClaimsHistory";

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="/analyze" element={<AnalyzeClaim />} />
            <Route path="/history" element={<ClaimsHistory />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}