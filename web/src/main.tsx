import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import { Shell } from "@/components/Shell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ALL_NAV } from "@/lib/nav";
import { NotFoundPage, PendingPage } from "@/pages/Pending";
import "./index.css";

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

const router = createBrowserRouter(
  [
    // pages load on demand: the landing page's animation libraries never ship with the terminal, and vice versa
    { index: true, lazy: () => import("@/pages/Landing").then((m) => ({ Component: m.LandingPage })) },
    {
      element: <Shell />,
      children: [
        { path: "/today", lazy: () => import("@/pages/Today").then((m) => ({ Component: m.TodayPage })) },
        { path: "/regimes/:symbol?", lazy: () => import("@/pages/Regimes").then((m) => ({ Component: m.RegimesPage })) },
        { path: "/volatility/:symbol?", lazy: () => import("@/pages/Volatility").then((m) => ({ Component: m.VolatilityPage })) },
        { path: "/states/:symbol?", lazy: () => import("@/pages/States").then((m) => ({ Component: m.StatesPage })) },
        { path: "/events/:symbol?", lazy: () => import("@/pages/Events").then((m) => ({ Component: m.EventsPage })) },
        { path: "/surface/:symbol?", lazy: () => import("@/pages/Surface").then((m) => ({ Component: m.SurfacePage })) },
        { path: "/risk", lazy: () => import("@/pages/Risk").then((m) => ({ Component: m.RiskPage })) },
        { path: "/us", lazy: () => import("@/pages/UsStocks").then((m) => ({ Component: m.UsStocksPage })) },
        { path: "/us/:ticker", lazy: () => import("@/pages/UsCompany").then((m) => ({ Component: m.UsCompanyPage })) },
        ...ALL_NAV.filter((n) => !n.ready).map((n) => ({ path: n.to, element: <PendingPage /> })),
        { path: "*", element: <NotFoundPage /> },
      ],
    },
  ],
  { basename: import.meta.env.BASE_URL.replace(/\/$/, "") || "/" },
);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={200}>
        <RouterProvider router={router} />
      </TooltipProvider>
    </QueryClientProvider>
  </StrictMode>,
);
