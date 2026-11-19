// The Workbench's lint rules: correctness, not layout. ESLint's and typescript-eslint's recommended
// sets, and eslint-plugin-vue's essential set (the recommended set adds layout rules this code has
// never followed; vue-tsc and the build already check types and templates).
import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist/**', 'coverage/**', 'playwright-report/**', 'test-results/**', 'node_modules/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  {
    files: ['**/*.vue'],
    languageOptions: { parserOptions: { parser: tseslint.parser } },
  },
  {
    files: ['**/*.ts', '**/*.vue'],
    // TypeScript resolves names itself; typescript-eslint advises this rule off for TS sources.
    rules: { 'no-undef': 'off' },
  },
  {
    // A leading underscore marks a name unused on purpose (a required parameter, a destructured rest).
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_', caughtErrorsIgnorePattern: '^_' },
      ],
    },
  },
  {
    // Unit tests stub Chart.js and other third-party shapes loosely.
    files: ['**/*.test.ts'],
    rules: { '@typescript-eslint/no-explicit-any': 'off' },
  },
  {
    // Playwright requires a hook's first argument to be a destructuring pattern, even an empty one.
    files: ['e2e/**/*.ts'],
    rules: { 'no-empty-pattern': 'off' },
  },
  {
    // Node scripts (corpus preparation for e2e, the platform root resolver).
    files: ['**/*.mjs'],
    languageOptions: { globals: { process: 'readonly', console: 'readonly' } },
  },
  {
    // Older eslint-disable comments name rules this config does not turn on.
    linterOptions: { reportUnusedDisableDirectives: 'off' },
  },
)
