import { ChangeDetectionStrategy, Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { DataService } from './data.service';
import { CompareResponse, MODEL_STYLE, Reading } from './models';

interface Card {
  key: string; label: string; color: string;
  intact: Reading; degraded: Reading; flipped: boolean; delta: number;
}

@Component({
  selector: 'app-predict',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
  @if (features().length) {
  <div class="split">
    <section class="panel">
      <div class="panel-head">
        <h2>Detector sensors</h2>
        <label class="inline">Start from
          <select (change)="loadExample(+$any($event.target).value)">
            @for (s of data.info()!.samples; track s.label; let i = $index) { <option [value]="i" [selected]="i === exampleIdx()">{{ s.label }}</option> }
          </select>
        </label>
      </div>

      <div class="sensor-grid" role="group" aria-label="The 50 detector sensors">
        @for (f of features(); track f.name; let i = $index) {
          <button type="button" class="tile" [class.off]="offline()[i]" [class.top]="f.rank <= 5"
                  [attr.aria-pressed]="offline()[i]"
                  [attr.aria-label]="'Sensor ' + (i + 1) + ', ' + f.name + ', importance rank ' + f.rank + ', ' + (offline()[i] ? 'offline' : 'live')"
                  [attr.title]="f.name + ' · importance rank ' + f.rank + ' of ' + features().length"
                  (click)="toggle(i)">
            <span class="n">{{ i + 1 }}</span>
            <span class="imp" [style.height.%]="impPct(f.importance)"></span>
          </button>
        }
      </div>
      <p class="legend">
        <span class="key key-top"></span> 5 most important sensors
        <span class="key key-off"></span> offline (reads 0)
        <span class="dim">Tile bar height = importance. Click a tile to switch that sensor off.</span>
      </p>

      <div class="actions" role="group" aria-label="Failure scenarios">
        <span class="actions-label">Switch off</span>
        @for (p of [10, 20, 30, 40]; track p) { <button type="button" class="btn" (click)="failRandom(p / 100)">{{ p }}% at random</button> }
        <button type="button" class="btn" (click)="failRank(true)">Top 5</button>
        <button type="button" class="btn" (click)="failRank(false)">Bottom 5</button>
        <button type="button" class="btn ghost" (click)="restore()" [disabled]="!nOffline()">Restore all</button>
      </div>

      <details class="raw">
        <summary>Edit raw sensor values</summary>
        <div class="raw-grid">
          @for (f of features(); track f.name; let i = $index) {
            <label class="raw-row" [class.off]="offline()[i]">
              <span class="raw-name">{{ i + 1 }} <small>{{ f.name }}</small></span>
              <input type="number" step="any" [value]="text()[i]" (input)="setText(i, $event)" [attr.aria-invalid]="!isNum(text()[i])" />
              <input type="checkbox" [checked]="offline()[i]" (change)="toggle(i)" [attr.aria-label]="'Sensor ' + (i + 1) + ' offline'" />
            </label>
          }
        </div>
      </details>
    </section>

    <aside class="panel results" aria-live="polite">
      <div class="panel-head">
        <h2>Baseline vs dropout-trained</h2>
        <span class="count"><strong>{{ nOffline() }}</strong> of {{ features().length }} offline</span>
      </div>

      @switch (data.apiState()) {
        @case ('static') {
          <p class="note">This public page has no prediction server attached, so it cannot score readings. The sensor grid still works as a preview; the <strong>Study results</strong> tab shows the measured comparison. To run predictions, start the Flask API locally (see the README).</p>
        }
        @case ('offline') {
          @if (data.health() && !data.health()!.model_loaded) {
            <p class="note warn">Flask is running but has no trained model. Run <code>python backend/train.py</code>, restart Flask, then check again.</p>
          } @else {
            <p class="note warn">The prediction API is not reachable. Start Flask (<code>python backend/app.py</code>) and run <code>npm start</code> in <code>frontend/</code>, then check again.</p>
          }
          <button type="button" class="btn" (click)="data.checkApi()">Check again</button>
        }
        @case ('checking') { <p class="note">Connecting to the API…</p> }
        @default {
          @if (data.mismatch()) {
            <p class="note warn">The API is serving a different model than the one this form was built from. Rebuild the frontend after training, or restart Flask.</p>
          } @else if (!allNumeric()) {
            <p class="note warn">Every sensor needs a number. Fix the highlighted field in “Edit raw sensor values”.</p>
          } @else if (error()) {
            <p class="note warn" role="alert">{{ error() }}</p>
          }
          @if (summary(); as s) { <p class="summary">{{ s }}</p> }
          <div class="cards">
            @for (c of cards(); track c.key) {
              <article class="card" [style.--c]="c.color">
                <h3>{{ c.label }}</h3>
                <div class="row">
                  <span class="row-label">All sensors live</span>
                  <span class="verdict">{{ verdict(c.intact) }}</span>
                  <div class="bar" role="img" [attr.aria-label]="'Probability of signal ' + pct(c.intact.probability_signal)">
                    <i [style.width.%]="c.intact.probability_signal * 100"></i><b></b>
                  </div>
                  <span class="pct">{{ pct(c.intact.probability_signal) }}</span>
                </div>
                @if (nOffline() > 0) {
                  <div class="row">
                    <span class="row-label">{{ nOffline() }} offline</span>
                    <span class="verdict">{{ verdict(c.degraded) }}</span>
                    <div class="bar" role="img" [attr.aria-label]="'Probability of signal ' + pct(c.degraded.probability_signal)">
                      <i [style.width.%]="c.degraded.probability_signal * 100"></i><b></b>
                    </div>
                    <span class="pct">{{ pct(c.degraded.probability_signal) }}</span>
                  </div>
                  <p class="foot" [class.flip]="c.flipped">
                    {{ c.flipped ? 'Answer changed' : 'Answer held' }} · confidence in signal {{ signed(c.delta) }} points
                  </p>
                }
              </article>
            }
          </div>
          @if (!cards().length && !error() && allNumeric()) { <p class="note">{{ loading() ? 'Scoring…' : 'Waiting for the first reading.' }}</p> }
          <p class="fine">Bars show the model’s probability that the event is a signal (electron neutrino); the tick marks 50%. One reading is an anecdote: the Study results tab has the statistics over the whole test set.</p>
        }
      }
    </aside>
  </div>
  }
  `,
})
export class PredictComponent {
  protected data = inject(DataService);

  protected features = computed(() => this.data.info()?.features ?? []);
  protected text = signal<string[]>([]);
  protected offline = signal<boolean[]>([]);
  protected exampleIdx = signal(0);
  protected result = signal<CompareResponse | null>(null);
  protected loading = signal(false);
  protected error = signal('');

  protected nums = computed(() => this.text().map((t) => (t.trim() === '' ? NaN : Number(t))));
  protected allNumeric = computed(() => this.nums().length > 0 && this.nums().every(Number.isFinite));
  protected nOffline = computed(() => this.offline().filter(Boolean).length);
  private maxImp = computed(() => Math.max(1e-12, ...this.features().map((f) => f.importance)));
  private reqId = 0;

  protected cards = computed<Card[]>(() => {
    const r = this.result();
    if (!r) return [];
    return [['baseline', 'xgb_baseline'], ['robust', 'xgb_robust']].filter(([k]) => r.models[k]).map(([k, style]) => {
      const m = r.models[k];
      return {
        key: k, label: MODEL_STYLE[style].label, color: MODEL_STYLE[style].color,
        intact: m.intact, degraded: m.degraded,
        flipped: m.intact.prediction !== m.degraded.prediction,
        delta: (m.degraded.probability_signal - m.intact.probability_signal) * 100,
      };
    });
  });

  protected summary = computed(() => {
    const cs = this.cards();
    if (cs.length < 2) return '';
    const n = this.nOffline();
    if (n === 0) return 'All sensors are live. Switch some off to see how each model reacts.';
    const [b, r] = cs;
    if (!b.flipped && !r.flipped) return `With ${n} sensors off, both models kept their answer.`;
    if (b.flipped && !r.flipped) return `With ${n} sensors off, the baseline changed its answer; the dropout-trained model held.`;
    if (!b.flipped && r.flipped) return `With ${n} sensors off, the dropout-trained model changed its answer; the baseline held.`;
    return `With ${n} sensors off, both models changed their answer.`;
  });

  constructor() {
    effect(() => {
      const f = this.features();
      if (f.length && untracked(() => this.text().length) !== f.length) untracked(() => this.loadExample(0));
    });
    // Re-score shortly after any change (live preview while sensors are toggled).
    effect((onCleanup) => {
      this.text(); this.offline();
      if (this.data.apiState() !== 'online' || this.data.mismatch() || !this.allNumeric()) return;
      const t = setTimeout(() => this.run(), 250);
      onCleanup(() => clearTimeout(t));
    });
  }

  protected impPct(v: number) { return Math.max(8, (v / this.maxImp()) * 100); }
  protected isNum(t: string) { return t.trim() !== '' && Number.isFinite(Number(t)); }
  protected pct(p: number) { return `${(p * 100).toFixed(1)}%`; }
  protected signed(v: number) { return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`; }
  protected verdict(r: Reading) { return r.prediction === 1 ? 'signal' : 'background'; }

  protected loadExample(i: number) {
    const s = this.data.info()?.samples[i];
    if (!s) return;
    this.exampleIdx.set(i);
    this.text.set(s.values.map(String));
    this.offline.set(this.features().map(() => false));
    this.result.set(null);
  }
  protected setText(i: number, ev: Event) {
    const v = (ev.target as HTMLInputElement).value;
    this.text.update((t) => t.map((x, j) => (j === i ? v : x)));
  }
  protected toggle(i: number) { this.offline.update((o) => o.map((x, j) => (j === i ? !x : x))); }
  protected restore() { this.offline.set(this.features().map(() => false)); }
  protected failRandom(rate: number) {
    const n = this.features().length, k = Math.max(1, Math.round(rate * n));
    const idx = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const chosen = new Set(idx.slice(0, k));
    this.offline.set(this.features().map((_, i) => chosen.has(i)));
  }
  protected failRank(top: boolean) {
    const n = this.features().length;
    this.offline.set(this.features().map((f) => (top ? f.rank <= 5 : f.rank > n - 5)));
  }

  private run() {
    const f = this.features(), nums = this.nums(), off = this.offline();
    const features: Record<string, number> = {};
    f.forEach((x, i) => (features[x.name] = nums[i]));
    const id = ++this.reqId;
    this.loading.set(true);
    this.data.compare(features, f.filter((_, i) => off[i]).map((x) => x.name)).subscribe({
      next: (r) => { if (id === this.reqId) { this.result.set(r); this.error.set(''); this.loading.set(false); } },
      error: (e) => {
        if (id !== this.reqId) return;
        this.loading.set(false);
        this.error.set(`API error ${e.status || ''}: ${e.error?.error || e.message || 'check the Flask terminal'}`);
      },
    });
  }
}
