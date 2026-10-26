import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export interface ChartSeries { key: string; label: string; color: string; mean: number[]; std: number[]; dashed?: boolean; }

/** Small dependency-free SVG line chart with +/- 1 sd error bars. Values are fractions (0..1) shown as %. */
@Component({
  selector: 'app-line-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" role="img" [attr.aria-label]="title()">
      @for (t of yTicks(); track t.v) {
        <line [attr.x1]="L" [attr.x2]="W - R" [attr.y1]="t.y" [attr.y2]="t.y" class="grid" />
        <text [attr.x]="L - 8" [attr.y]="t.y + 4" text-anchor="end" class="tick">{{ t.label }}</text>
      }
      @for (t of xTicks(); track t.i) {
        <text [attr.x]="t.x" [attr.y]="H - B + 20" text-anchor="middle" class="tick">{{ t.label }}</text>
      }
      <text [attr.x]="(L + W - R) / 2" [attr.y]="H - 6" text-anchor="middle" class="axis">{{ xLabel() }}</text>
      <text [attr.x]="14" [attr.y]="(T + H - B) / 2" text-anchor="middle" class="axis"
            [attr.transform]="'rotate(-90 14 ' + (T + H - B) / 2 + ')'">{{ yLabel() }}</text>
      @for (s of paths(); track s.key) {
        @for (b of s.bars; track b.x) {
          <line [attr.x1]="b.x" [attr.x2]="b.x" [attr.y1]="b.y1" [attr.y2]="b.y2" [attr.stroke]="s.color" stroke-width="1.5" opacity=".45" />
        }
        <polyline [attr.points]="s.points" fill="none" [attr.stroke]="s.color" stroke-width="2.4"
                  [attr.stroke-dasharray]="s.dashed ? '6 4' : null" stroke-linejoin="round" />
        @for (d of s.dots; track d.x) {
          <circle [attr.cx]="d.x" [attr.cy]="d.y" r="3.6" [attr.fill]="s.color" />
        }
      }
    </svg>
  `,
})
export class LineChartComponent {
  readonly W = 640; readonly H = 330; readonly L = 52; readonly R = 16; readonly T = 14; readonly B = 52;

  title = input('Line chart');
  xs = input.required<number[]>();
  series = input.required<ChartSeries[]>();
  xLabel = input('');
  yLabel = input('Accuracy (%)');
  xFormat = input<(v: number) => string>((v) => String(v));

  private yRange = computed(() => {
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series()) s.mean.forEach((m, i) => {
      lo = Math.min(lo, m - (s.std[i] ?? 0)); hi = Math.max(hi, m + (s.std[i] ?? 0));
    });
    if (!isFinite(lo)) { lo = 0.5; hi = 1; }
    const span = Math.max(hi - lo, 0.04);
    const step = span > 0.4 ? 0.1 : span > 0.15 ? 0.05 : span > 0.06 ? 0.02 : 0.01;
    return { lo: Math.floor((lo - 0.01) / step) * step, hi: Math.min(1, Math.ceil((hi + 0.01) / step) * step), step };
  });

  private px = (i: number) => {
    const n = this.xs().length;
    return this.L + (n <= 1 ? 0 : (i / (n - 1)) * (this.W - this.L - this.R));
  };
  private py = (v: number) => {
    const { lo, hi } = this.yRange();
    return this.T + (1 - (v - lo) / (hi - lo || 1)) * (this.H - this.T - this.B);
  };

  yTicks = computed(() => {
    const { lo, hi, step } = this.yRange(); const out: { v: number; y: number; label: string }[] = [];
    for (let v = lo; v <= hi + 1e-9; v += step) out.push({ v, y: this.py(v), label: (v * 100).toFixed(step < 0.02 ? 1 : 0) });
    return out;
  });
  xTicks = computed(() => this.xs().map((v, i) => ({ i, x: this.px(i), label: this.xFormat()(v) })));

  paths = computed(() => this.series().map((s) => ({
    key: s.key, color: s.color, dashed: s.dashed,
    points: s.mean.map((m, i) => `${this.px(i)},${this.py(m)}`).join(' '),
    dots: s.mean.map((m, i) => ({ x: this.px(i), y: this.py(m) })),
    bars: s.mean.map((m, i) => ({ x: this.px(i), y1: this.py(m - (s.std[i] ?? 0)), y2: this.py(m + (s.std[i] ?? 0)) })),
  })));
}
