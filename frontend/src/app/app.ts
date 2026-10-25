import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { DataService } from './data.service';
import { PredictComponent } from './predict.component';
import { ResearchComponent } from './research.component';

type Tab = 'predict' | 'study' | 'about';

@Component({
  selector: 'app-root',
  imports: [PredictComponent, ResearchComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './app.html',
})
export class App {
  protected data = inject(DataService);
  protected tab = signal<Tab>('predict');
  protected tabs: { id: Tab; label: string }[] = [
    { id: 'predict', label: 'Test a reading' },
    { id: 'study', label: 'Study results' },
    { id: 'about', label: 'About' },
  ];

  protected status = computed(() => {
    switch (this.data.apiState()) {
      case 'online': return { cls: 'ok', text: this.data.health()?.synthetic ? 'API online · demo model' : 'API online' };
      case 'offline': return { cls: 'bad', text: 'API offline' };
      case 'static': return { cls: 'idle', text: 'Static preview · no API' };
      default: return { cls: 'idle', text: 'Connecting…' };
    }
  });

  constructor() { void this.data.init(); }
}
