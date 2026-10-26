import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { DataService } from './data.service';
import { ChartSeries, LineChartComponent } from './line-chart.component';
import { MODEL_STYLE, Series } from './models';

const ORDER = ['xgb_baseline', 'xgb_robust', 'xgb_baseline_mean_impute', 'mlp'];

function toSeries(src: Record<string, Series>, keys = ORDER): ChartSeries[] {
  return keys.filter((k) => src[k]).map((k) => ({
    key: k, label: MODEL_STYLE[k].label, color: MODEL_STYLE[k].color, mean: src[k].mean, std: src[k].std,
    dashed: k === 'xgb_baseline_mean_impute',
  }));
}
const p = (v: number, d = 1) => (v * 100).toFixed(d);

@Component({
  selector: 'app-research',
  imports: [LineChartComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
  @if (r(); as res) {
    @if (res.synthetic) {
      <p class="note warn banner">These numbers come from <strong>synthetic demo data</strong>, not MiniBooNE. Run <code>python backend/train.py</code>, rebuild, and they are replaced by the real results.</p>
    }

    <section class="panel">
      <h2>1. Baseline replication</h2>
      <p class="lede">{{ res.n_rows.toLocaleString() }} events × {{ res.n_features }} features, {{ res.split }}. Train {{ res.n_train.toLocaleString() }} / test {{ res.n_test.toLocaleString() }}.</p>
      <table class="tbl">
        <thead><tr><th>Model</th><th class="num">Test accuracy</th><th class="num">ROC AUC</th></tr></thead>
        <tbody>
          @for (m of res.replication; track m.model) {
            <tr><td><i class="dot" [style.background]="color(m.model)"></i>{{ label(m.model) }}</td><td class="num">{{ pc(m.accuracy, 2) }}%</td><td class="num">{{ m.auc.toFixed(4) }}</td></tr>
          }
        </tbody>
      </table>
      <p class="fine">Compare these with the table in the benchmark paper you replicate, and state any protocol differences (split ratio, tuning budget, preprocessing).</p>
    </section>

    <section class="panel">
      <h2>2. Sensor dropout</h2>
      <p class="lede">A random subset of sensors reads 0 for the whole test set. Mean ± 1 sd over {{ res.dropout.seeds }} random subsets.</p>
      <div class="chart-wrap">
        <app-line-chart title="Accuracy versus percentage of sensors dropped" [xs]="res.dropout.rates" [series]="dropoutSeries()" xLabel="Sensors switched off" [xFormat]="pctFmt" />
        <ul class="legend-list">@for (s of dropoutSeries(); track s.key) { <li><i class="dot" [style.background]="s.color"></i>{{ s.label }}</li> }</ul>
      </div>
      @if (dropoutFact(); as f) { <p class="finding">{{ f }}</p> }
    </section>

    <section class="panel">
      <h2>3. Targeted noise</h2>
      <p class="lede">Gaussian noise with σ = multiple of each feature’s training std, added to the {{ res.noise.k }} most important versus the {{ res.noise.k }} least important sensors (ranked by XGBoost gain on the training set).</p>
      <div class="two">
        <figure>
          <figcaption>{{ res.noise.k }} most important sensors</figcaption>
          <app-line-chart title="Accuracy under noise on the most important sensors" [xs]="res.noise.alphas" [series]="noiseSeries('top')" xLabel="Noise σ" [xFormat]="sigmaFmt" />
        </figure>
        <figure>
          <figcaption>{{ res.noise.k }} least important sensors</figcaption>
          <app-line-chart title="Accuracy under noise on the least important sensors" [xs]="res.noise.alphas" [series]="noiseSeries('bottom')" xLabel="Noise σ" [xFormat]="sigmaFmt" />
        </figure>
      </div>
      <ul class="legend-list">@for (s of noiseSeries('top'); track s.key) { <li><i class="dot" [style.background]="s.color"></i>{{ s.label }}</li> }</ul>
      @if (noiseFact(); as f) { <p class="finding">{{ f }}</p> }
    </section>

    <section class="panel">
      <h2>How to read these results</h2>
      <ul class="plain">
        <li>The dropout-trained model sees the same <em>kind</em> of failure it is tested on, so dropout curves favour it by construction. The noise experiment is the fairer test: it never trained on noise.</li>
        <li>The mean-fill line assumes the system knows which sensors died. It is an optimistic reference, not a like-for-like rival.</li>
        <li>Zero is one way a sensor can fail. Stuck-at-last-value or drifting readings behave differently and are not covered.</li>
        <li>Accuracy on a nearly balanced test set is the only metric shown here; add precision/recall or AUC under corruption if the report needs them.</li>
      </ul>
    </section>
  } @else {
    <p class="note warn">{{ data.loadError() || 'Loading results…' }}</p>
  }
  `,
})
export class ResearchComponent {
  protected data = inject(DataService);
  protected r = this.data.results;
  protected pc = p;
  protected pctFmt = (v: number) => `${Math.round(v * 100)}%`;
  protected sigmaFmt = (v: number) => `${v}σ`;

  protected label(k: string) { return MODEL_STYLE[k]?.label ?? k; }
  protected color(k: string) { return MODEL_STYLE[k]?.color ?? '#888'; }

  protected dropoutSeries = computed(() => toSeries(this.r()?.dropout.series ?? {}));
  protected noiseSeries(group: string) { return toSeries(this.r()?.noise.groups[group]?.series ?? {}, ORDER.slice(0, 2).concat('mlp')); }

  protected dropoutFact = computed(() => {
    const res = this.r(); if (!res) return '';
    const b = res.dropout.series['xgb_baseline'], g = res.dropout.series['xgb_robust'];
    if (!b || !g) return '';
    const last = res.dropout.rates.length - 1, rate = Math.round(res.dropout.rates[last] * 100);
    const gap = (g.mean[last] - b.mean[last]) * 100, cost = (g.mean[0] - b.mean[0]) * 100;
    return `With ${rate}% of sensors off, the baseline falls from ${p(b.mean[0])}% to ${p(b.mean[last])}% and the dropout-trained model from ${p(g.mean[0])}% to ${p(g.mean[last])}%: `
      + `${Math.abs(gap).toFixed(1)} points ${gap >= 0 ? 'better' : 'worse'} for the dropout-trained model. On clean data it is ${Math.abs(cost).toFixed(1)} points ${cost >= 0 ? 'higher' : 'lower'}.`;
  });

  protected noiseFact = computed(() => {
    const res = this.r(); if (!res) return '';
    const top = res.noise.groups['top'], bot = res.noise.groups['bottom'];
    const b = top?.series['xgb_baseline'], g = top?.series['xgb_robust'], bb = bot?.series['xgb_baseline'];
    if (!b || !g || !bb) return '';
    const last = res.noise.alphas.length - 1, a = res.noise.alphas[last];
    const th = top.collapse_alpha['xgb_baseline'];
    return `At σ = ${a}, noise on the most important sensors takes the baseline to ${p(b.mean[last])}% and the dropout-trained model to ${p(g.mean[last])}%; `
      + `the same noise on the least important sensors leaves the baseline at ${p(bb.mean[last])}%. `
      + (th !== null && th !== undefined ? `The baseline first loses 5 points at σ = ${th}.` : 'The baseline never loses 5 points in the tested range.');
  });
}
