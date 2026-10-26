import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, firstValueFrom } from 'rxjs';
import { ApiState, AppConfig, CompareResponse, Health, ModelInfo, Results } from './models';

@Injectable({ providedIn: 'root' })
export class DataService {
  private http = inject(HttpClient);

  readonly config = signal<AppConfig>({ apiBase: '' });
  readonly info = signal<ModelInfo | null>(null);
  readonly results = signal<Results | null>(null);
  readonly loadError = signal('');
  readonly apiState = signal<ApiState>('checking');
  readonly health = signal<Health | null>(null);

  /** True when the running API serves a different model than the one the form was built from. */
  mismatch(): boolean {
    const h = this.health(), i = this.info();
    return !!(h && i && h.model_id && h.model_id !== i.model_id);
  }

  async init(): Promise<void> {
    try {
      // Relative URLs keep working under a GitHub Pages sub-path (<base href>).
      this.config.set(await firstValueFrom(this.http.get<AppConfig>('config.json')));
      this.info.set(await firstValueFrom(this.http.get<ModelInfo>('model_info.json')));
      this.results.set(await firstValueFrom(this.http.get<Results>('results.json')));
    } catch {
      this.loadError.set('Could not load model_info.json / results.json. Run: python backend/train.py');
    }
    await this.checkApi();
  }

  async checkApi(): Promise<void> {
    const base = this.config().apiBase;
    if (!base) { this.apiState.set('static'); return; }
    this.apiState.set('checking');
    try {
      const h = await firstValueFrom(this.http.get<Health>(`${base}/health`));
      this.health.set(h);
      this.apiState.set(h.model_loaded ? 'online' : 'offline');
    } catch {
      this.apiState.set('offline');
    }
  }

  compare(features: Record<string, number>, offline: string[]): Observable<CompareResponse> {
    return this.http.post<CompareResponse>(`${this.config().apiBase}/compare`, { features, offline });
  }
}
