// cytoscape-fcose ships no TypeScript types. It's a standard cytoscape extension:
// a function passed to `cytoscape.use(...)`. Minimal ambient declaration so
// vue-tsc strict accepts the import + registration.
declare module 'cytoscape-fcose' {
  import type cytoscape from 'cytoscape'
  const fcose: (cy: typeof cytoscape) => void
  export default fcose
}
