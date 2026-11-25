/**
 * Live prices fan-out: while at least one signed-in viewer is connected, poll the quant service's /ticks (fast while
 * NSE is open, slow when closed) and push each new payload to every viewer. No viewers, no polling, no provider calls.
 */
export type Send = (payload: string) => void;
type Log = { warn: (obj: object, msg: string) => void };

export class TickHub {
  private clients = new Set<Send>();
  private timer: NodeJS.Timeout | undefined;
  private last: string | undefined;
  private stopped = false;

  constructor(
    private load: () => Promise<string>,
    private intervals: { openMs: number; closedMs: number; retryMs: number },
    private log?: Log,
  ) {}

  get size(): number {
    return this.clients.size;
  }

  subscribe(send: Send): () => void {
    this.clients.add(send);
    if (this.last) send(this.last);
    if (this.clients.size === 1) void this.tick();
    return () => {
      this.clients.delete(send);
      if (this.clients.size === 0) this.pause();
    };
  }

  close(): void {
    this.stopped = true;
    this.pause();
    this.clients.clear();
  }

  private pause(): void {
    clearTimeout(this.timer);
    this.timer = undefined;
  }

  private async tick(): Promise<void> {
    if (this.stopped || this.clients.size === 0) return;
    let next = this.intervals.retryMs;
    try {
      const body = await this.load();
      const open = (JSON.parse(body) as { market_open?: boolean }).market_open === true;
      next = open ? this.intervals.openMs : this.intervals.closedMs;
      if (body !== this.last) {
        this.last = body;
        for (const send of this.clients) send(body);
      }
    } catch (err) {
      this.log?.warn({ err }, "live prices unavailable; retrying");
    }
    if (!this.stopped && this.clients.size > 0) {
      clearTimeout(this.timer);
      this.timer = setTimeout(() => void this.tick(), next);
    }
  }
}
