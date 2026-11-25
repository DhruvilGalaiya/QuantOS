import type { JobRun, JobSummary, System } from "@quantos/types";
import { useEffect } from "react";
import { CineHero, CineSection, Method, Spot } from "@/components/cine";
import { ErrorBox, Skeleton } from "@/components/kit";
import { useRoute } from "@/lib/api";
import { DASH, day, stamp } from "@/lib/format";
import { cn } from "@/lib/utils";

/** "3 h ago" measured against the page's own clock (when the data was exported), so it stays pure. */
function ago(iso: string | null | undefined, now: string): string {
  if (!iso) return "never";
  const s = (new Date(now).getTime() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(s)) return DASH;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400 * 1.5) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

function seconds(x: number | null | undefined): string {
  if (x == null) return DASH;
  return x < 90 ? `${Math.round(x)} s` : `${(x / 60).toFixed(1)} min`;
}

function Dot({ ok }: { ok: boolean | null | undefined }) {
  return <span className={cn("inline-block size-2 shrink-0 rounded-full", ok == null ? "bg-faint" : ok ? "bg-up" : "bg-down")} aria-hidden />;
}

const word = (ok: boolean | null | undefined) => (ok == null ? "Not run yet" : ok ? "Worked" : "Failed");

function JobCard({ j, now }: { j: JobSummary; now: string }) {
  return (
    <Spot glow={j.last_ok === false ? "rgba(238, 94, 90, 0.45)" : undefined}>
      <div className="flex items-center gap-2.5">
        <Dot ok={j.last_ok} />
        <span className={cn("text-[13px]", j.last_ok === false ? "text-down" : "text-muted-foreground")}>{word(j.last_ok)}</span>
        <span className="ml-auto text-[13px] text-faint">{ago(j.last_run, now)}</span>
      </div>
      <h3 className="mt-4 text-[19px] font-semibold leading-snug text-foreground">{j.label}</h3>
      <p className="mt-1 text-[14px] text-muted-foreground">{j.schedule}</p>
      {j.last_ok === false && (
        <p className="mt-3 text-[13.5px] text-muted-foreground">
          Last good run {j.last_success ? stamp(j.last_success) : "not yet"}. {j.message && <span className="font-mono text-[12.5px] text-faint">{j.message}</span>}
        </p>
      )}
    </Spot>
  );
}

function RunsTable({ runs }: { runs: JobRun[] }) {
  return (
    <div className="mt-8 overflow-x-auto rounded-2xl ring-1 ring-line">
      <table className="w-full min-w-[620px] text-left text-[14px]">
        <thead className="text-[12.5px] text-faint">
          <tr className="border-b border-line">
            <th className="px-4 py-3 font-normal">Started</th>
            <th className="px-4 py-3 font-normal">Job</th>
            <th className="px-4 py-3 font-normal">Result</th>
            <th className="px-4 py-3 text-right font-normal">Took</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={`${r.job}-${r.started}-${i}`} className="border-b border-line/60 last:border-0">
              <td className="num whitespace-nowrap px-4 py-2.5 text-muted-foreground">{stamp(r.started)}</td>
              <td className="px-4 py-2.5 text-foreground">{r.label}</td>
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-2">
                  <Dot ok={r.ok} />
                  <span className={r.ok === false ? "text-down" : "text-muted-foreground"}>{r.ok === false ? r.message || "Failed" : word(r.ok)}</span>
                </span>
              </td>
              <td className="num px-4 py-2.5 text-right text-muted-foreground">{seconds(r.duration_s)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function headline(d: System): { title: string; failing: JobSummary[] } {
  const ran = d.jobs.filter((j) => j.last_ok != null);
  const failing = ran.filter((j) => j.last_ok === false);
  if (!ran.length) return { title: "No refresh has run yet.", failing };
  if (!failing.length) return { title: "Every job worked on its last run.", failing };
  return { title: `${failing.length} of ${ran.length} jobs failed on their last run.`, failing };
}

export function StatusPage() {
  const { data, error, isPending } = useRoute("system");
  useEffect(() => {
    document.title = "Data status · QuantOS";
  }, []);

  if (isPending) return <Skeleton className="h-[60vh]" />;
  if (error) return <ErrorBox title="The data status could not be loaded">{error.message}. Reload the page to try again.</ErrorBox>;

  const { title, failing } = headline(data);
  const prices = data.freshness.find((f) => f.dataset === "india_daily");
  const latest = data.runs[0];

  return (
    <>
      <CineHero
        glow={failing.length ? "rgb(238 94 90 / 0.22)" : "rgb(76 195 160 / 0.20)"}
        title={title}
        lede={
          <>
            QuantOS refreshes itself on GitHub's servers every weekday: index prices and option chains come in, the models run, and this site
            rebuilds. {failing.length ? "Until the next good run, every page keeps showing the last good data." : "Nobody has to press a button."}
          </>
        }
      >
        <dl className="grid max-w-3xl grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-3">
          <div>
            <dt className="text-[13px] text-muted-foreground">Last refresh</dt>
            <dd className="headline mt-1 text-[clamp(1.6rem,1.2rem+1.4vw,2.4rem)]">{ago(latest?.started, data.now)}</dd>
          </div>
          <div>
            <dt className="text-[13px] text-muted-foreground">Index prices through</dt>
            <dd className="headline mt-1 text-[clamp(1.6rem,1.2rem+1.4vw,2.4rem)]">{prices ? day(prices.as_of) : DASH}</dd>
          </div>
          <div>
            <dt className="text-[13px] text-muted-foreground">Recent runs that worked</dt>
            <dd className="headline mt-1 text-[clamp(1.6rem,1.2rem+1.4vw,2.4rem)]">
              {data.runs.filter((r) => r.ok).length}
              <span className="text-muted-foreground"> of {data.runs.length}</span>
            </dd>
          </div>
        </dl>
      </CineHero>

      <CineSection id="jobs" title="Four jobs keep the data current" lede="Each runs on a schedule in India time. GitHub can start a run a little late when it is busy.">
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {data.jobs.map((j) => (
            <JobCard key={j.job} j={j} now={data.now} />
          ))}
        </div>
      </CineSection>

      <CineSection id="fresh" title="How fresh each dataset is" lede="Market time each dataset covers, and when it was last written.">
        <ul className="mt-8 divide-y divide-line/70 rounded-2xl ring-1 ring-line">
          {data.freshness.map((f) => (
            <li key={f.dataset} className="grid gap-1 px-5 py-4 sm:grid-cols-[minmax(0,1fr)_auto_auto] sm:items-baseline sm:gap-8">
              <span className="text-[15px] text-foreground">{f.label}</span>
              <span className="num text-[14px] text-muted-foreground">covers {f.as_of.length > 10 ? stamp(f.as_of) : day(f.as_of)}</span>
              <span className="text-[13px] text-faint">written {ago(f.updated, data.now)}</span>
            </li>
          ))}
        </ul>
      </CineSection>

      <CineSection id="runs" title="Recent runs" lede="Newest first. A failed run leaves the previous data in place.">
        {data.runs.length ? <RunsTable runs={data.runs} /> : <p className="mt-8 text-muted-foreground">No runs recorded yet.</p>}
        <Method>
          <p>
            A scheduled workflow on GitHub Actions restores the newest data, runs the jobs, saves the data back as one file on the repository's
            data-latest release, and rebuilds this site. Index prices and option chains come from Upstox, US prices from Yahoo Finance.
          </p>
          <p>Times on this page are measured from when the site was last built.</p>
        </Method>
      </CineSection>
    </>
  );
}
