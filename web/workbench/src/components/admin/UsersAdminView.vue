<script setup lang="ts">
/**
 * Admin → Users (#1128) — list every platform user (player + viewer) and manage them: change role,
 * activate/deactivate, create, delete. Admin-only (the backend 403s otherwise; the tab is hidden for
 * non-admins). Self-affecting controls are disabled to match the backend's self-lockout guards.
 */
import { computed, onMounted, ref } from 'vue'
import {
  createUser,
  deleteUser,
  listUsers,
  patchUser,
  type AdminUser,
  type Role,
} from '../../api/authApi'
import { useAuthStore } from '../../stores/auth'

const auth = useAuthStore()
const users = ref<AdminUser[]>([])
const loading = ref(true)
const error = ref('')
const busy = ref<Set<string>>(new Set())

const ROLES: Role[] = ['listener', 'creator', 'admin']

const newEmail = ref('')
const newName = ref('')
const newRole = ref<Role>('creator')
const creating = ref(false)

const sorted = computed(() =>
  [...users.value].sort((a, b) => a.email.localeCompare(b.email)),
)

function isSelf(u: AdminUser): boolean {
  return u.user_id === auth.user?.user_id
}

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    users.value = await listUsers()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load users'
  } finally {
    loading.value = false
  }
}

function mark(id: string, on: boolean): void {
  const next = new Set(busy.value)
  if (on) next.add(id)
  else next.delete(id)
  busy.value = next
}

async function changeRole(u: AdminUser, role: Role): Promise<void> {
  if (role === u.role) return
  mark(u.user_id, true)
  error.value = ''
  try {
    const updated = await patchUser(u.user_id, { role })
    users.value = users.value.map((x) => (x.user_id === u.user_id ? updated : x))
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to change role'
  } finally {
    mark(u.user_id, false)
  }
}

async function toggleActive(u: AdminUser): Promise<void> {
  mark(u.user_id, true)
  error.value = ''
  try {
    const updated = await patchUser(u.user_id, { disabled: !u.disabled })
    users.value = users.value.map((x) => (x.user_id === u.user_id ? updated : x))
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to update'
  } finally {
    mark(u.user_id, false)
  }
}

async function remove(u: AdminUser): Promise<void> {
  if (!window.confirm(`Delete ${u.email}? This cannot be undone.`)) return
  mark(u.user_id, true)
  error.value = ''
  try {
    await deleteUser(u.user_id)
    users.value = users.value.filter((x) => x.user_id !== u.user_id)
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to delete'
  } finally {
    mark(u.user_id, false)
  }
}

async function create(): Promise<void> {
  if (!newEmail.value.trim()) return
  creating.value = true
  error.value = ''
  try {
    const u = await createUser({
      email: newEmail.value.trim(),
      name: newName.value.trim(),
      role: newRole.value,
    })
    users.value = [...users.value, u]
    newEmail.value = ''
    newName.value = ''
    newRole.value = 'creator'
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to create user'
  } finally {
    creating.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="mx-auto max-w-4xl" data-testid="users-admin">
    <div class="mb-3 flex items-center justify-between">
      <h2 class="text-base font-semibold text-surface-foreground">Users</h2>
      <button
        type="button"
        class="rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay"
        @click="load"
      >
        Refresh
      </button>
    </div>

    <p
      v-if="error"
      class="mb-3 rounded border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger"
      data-testid="users-admin-error"
      role="alert"
    >
      {{ error }}
    </p>

    <!-- create -->
    <form
      class="mb-4 flex flex-wrap items-end gap-2 rounded border border-border bg-surface p-3"
      @submit.prevent="create"
    >
      <label class="flex flex-col text-xs text-muted">
        Email
        <input
          v-model="newEmail"
          type="email"
          required
          placeholder="user@example.com"
          class="mt-0.5 w-56 rounded border border-border bg-canvas px-2 py-1 text-sm text-canvas-foreground"
          data-testid="new-user-email"
        />
      </label>
      <label class="flex flex-col text-xs text-muted">
        Name
        <input
          v-model="newName"
          type="text"
          placeholder="(optional)"
          class="mt-0.5 w-40 rounded border border-border bg-canvas px-2 py-1 text-sm text-canvas-foreground"
        />
      </label>
      <label class="flex flex-col text-xs text-muted">
        Role
        <select
          v-model="newRole"
          class="mt-0.5 rounded border border-border bg-canvas px-2 py-1 text-sm text-canvas-foreground"
        >
          <option v-for="r in ROLES" :key="r" :value="r">{{ r }}</option>
        </select>
      </label>
      <button
        type="submit"
        :disabled="creating || !newEmail.trim()"
        class="rounded bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
        data-testid="create-user-button"
      >
        Add user
      </button>
    </form>

    <p v-if="loading" class="text-sm text-muted">Loading…</p>
    <table v-else class="w-full border-collapse text-sm">
      <thead>
        <tr class="border-b border-border text-left text-xs uppercase tracking-wide text-muted">
          <th class="py-2 pr-3 font-medium">User</th>
          <th class="py-2 pr-3 font-medium">Role</th>
          <th class="py-2 pr-3 font-medium">Status</th>
          <th class="py-2 font-medium"></th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="u in sorted"
          :key="u.user_id"
          class="border-b border-border/60"
          :data-testid="`user-row-${u.email}`"
        >
          <td class="py-2 pr-3">
            <div class="font-medium text-surface-foreground">{{ u.name || u.email }}</div>
            <div class="text-xs text-muted">{{ u.email }}<span v-if="isSelf(u)"> · you</span></div>
          </td>
          <td class="py-2 pr-3">
            <select
              :value="u.role"
              :disabled="isSelf(u) || busy.has(u.user_id)"
              class="rounded border border-border bg-canvas px-2 py-1 text-sm text-canvas-foreground disabled:opacity-50"
              :data-testid="`role-select-${u.email}`"
              @change="changeRole(u, ($event.target as HTMLSelectElement).value as Role)"
            >
              <option v-for="r in ROLES" :key="r" :value="r">{{ r }}</option>
            </select>
          </td>
          <td class="py-2 pr-3">
            <button
              type="button"
              :disabled="isSelf(u) || busy.has(u.user_id)"
              class="rounded-full px-2 py-0.5 text-xs font-medium disabled:opacity-50"
              :class="
                u.disabled
                  ? 'bg-danger/15 text-danger'
                  : 'bg-success/15 text-success'
              "
              :data-testid="`active-toggle-${u.email}`"
              @click="toggleActive(u)"
            >
              {{ u.disabled ? 'Inactive' : 'Active' }}
            </button>
          </td>
          <td class="py-2 text-right">
            <button
              type="button"
              :disabled="isSelf(u) || busy.has(u.user_id)"
              class="rounded border border-border px-2 py-1 text-xs text-muted hover:bg-overlay hover:text-danger disabled:opacity-50"
              :data-testid="`delete-user-${u.email}`"
              @click="remove(u)"
            >
              Delete
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
