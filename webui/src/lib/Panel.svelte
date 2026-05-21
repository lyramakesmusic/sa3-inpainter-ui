<script>
let { title, headerActions, children, defaultOpen = true } = $props();
let open = $state(defaultOpen);
</script>

<section class="panel">
  <header class="panel-header">
    <button class="title-btn" onclick={() => open = !open}>
      <i class="bi {open ? 'bi-chevron-down' : 'bi-chevron-right'} chev"></i>
      <span>{title}</span>
    </button>
    {#if headerActions}
      <div class="header-actions">{@render headerActions()}</div>
    {/if}
  </header>
  {#if open}
    <div class="panel-body">
      {@render children()}
    </div>
  {/if}
</section>

<style>
.panel { /* no hr — sections stand alone */ }
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--gap-3) var(--gap-4);
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
}
.title-btn {
  display: flex;
  align-items: center;
  gap: var(--gap-2);
  color: inherit;
  font: inherit;
  font-weight: inherit;
}
.title-btn:hover .chev { color: var(--text-primary); }
.chev {
  color: var(--text-secondary);
  font-size: 11px;
  transition: transform 0.12s;
}
.header-actions {
  display: flex;
  gap: var(--gap-1);
}
.panel-body {
  padding: 0 var(--gap-4) var(--gap-2);
  display: flex;
  flex-direction: column;
  gap: var(--gap-3);
}
</style>
