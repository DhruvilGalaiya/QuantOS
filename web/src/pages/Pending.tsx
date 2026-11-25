import { useLocation } from "react-router";
import { Link } from "react-router";
import { PageHeader, Panel } from "@/components/kit";
import { ALL_NAV } from "@/lib/nav";

/** Pages that move over from the research dashboard in later milestones. */
export function PendingPage() {
  const { pathname } = useLocation();
  const item = ALL_NAV.find((n) => n.to === pathname);
  if (!item) return <NotFoundPage />;
  return (
    <>
      <PageHeader title={item.label}>{item.blurb}.</PageHeader>
      <Panel className="max-w-2xl p-5 text-muted-foreground">
        This page is being rebuilt for the new terminal. Until it lands, the same analysis runs in the research dashboard (
        <code className="font-mono text-[12.5px]">python -m streamlit run backend/app/regime_dashboard_v10.py</code>).{" "}
        <Link to="/today" className="text-saffron underline underline-offset-2">
          Back to Today
        </Link>
      </Panel>
    </>
  );
}

export function NotFoundPage() {
  return (
    <>
      <PageHeader title="No page here">That address does not match a QuantOS page.</PageHeader>
      <Link to="/today" className="text-saffron underline underline-offset-2">
        Go to Today
      </Link>
    </>
  );
}
