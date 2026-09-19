import { accessSync, constants, realpathSync, statSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { execFile, spawn } from 'node:child_process';
const BUBBLEWRAP = '/usr/bin/bwrap';
const SYSTEMD_RUN = '/usr/bin/systemd-run';
const SYSTEMCTL = '/usr/bin/systemctl';
const SAFE_RUN_ID = /^[a-f0-9]{8,64}$/;
const SAFE_EXECUTABLE = /^(?:[A-Za-z0-9@._+-]+\/)*[A-Za-z0-9@._+-]+$/;
const SAFE_TOKEN = /^[a-f0-9]{64}$/;
const SAFE_MOUNT_TARGET = /^(?:[A-Za-z0-9._+-]+\/)*[A-Za-z0-9._+-]+$/;
export const RUN_DEADLINE_POLICY = Object.freeze({
    firstWarningSeconds: 60 * 60,
    finalWarningSeconds: 65 * 60,
    hardCutoffSeconds: 70 * 60,
});
export const DEFAULT_SANDBOX_LIMITS = Object.freeze({
    memoryBytes: 4 * 1024 * 1024 * 1024,
    cpuQuotaPercent: 200,
    tasksMax: 128,
    runtimeSeconds: RUN_DEADLINE_POLICY.hardCutoffSeconds,
    tmpBytes: 512 * 1024 * 1024,
});
export class SandboxConfigurationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'SandboxConfigurationError';
    }
}
function existingDirectory(path, label) {
    if (!isAbsolute(path) || path.includes('\0')) {
        throw new SandboxConfigurationError(`${label} must be an absolute path`);
    }
    let canonical;
    try {
        canonical = realpathSync(path);
        if (!statSync(canonical).isDirectory())
            throw new Error('not a directory');
    }
    catch {
        throw new SandboxConfigurationError(`${label} must reference an existing directory`);
    }
    return canonical;
}
function isWithin(parent, child) {
    const suffix = relative(parent, child);
    return (suffix === '' || (!suffix.startsWith(`..${sep}`) && suffix !== '..' && !isAbsolute(suffix)));
}
function boundedInteger(value, minimum, maximum, label) {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
        throw new SandboxConfigurationError(`${label} is outside the supported range`);
    }
    return value;
}
function resolveOptions(options) {
    if (!SAFE_RUN_ID.test(options.runId)) {
        throw new SandboxConfigurationError('runId must be 8–64 lowercase hexadecimal characters');
    }
    if (!SAFE_EXECUTABLE.test(options.runtimeExecutable)) {
        throw new SandboxConfigurationError('runtimeExecutable must be a safe relative path');
    }
    if (options.workspace.runId !== options.runId ||
        !/^[a-f0-9]{40}$/.test(options.workspace.revision) ||
        options.workspace.branch !== `agent/${options.runId}`) {
        throw new SandboxConfigurationError('workspace snapshot does not match the validated run');
    }
    const workspace = existingDirectory(options.workspace.path, 'workspace');
    const runtimeRoot = existingDirectory(options.runtimeRoot, 'runtimeRoot');
    const profileRoot = existingDirectory(options.profileRoot, 'profileRoot');
    if (options.writableProfile === true && dirname(profileRoot) !== dirname(workspace)) {
        throw new SandboxConfigurationError('writable profile must belong to the same private run as the workspace');
    }
    const optionalValues = [
        options.agentRuntimeRoot,
        options.agentExecutable,
        options.inferenceSocketRoot,
        options.inferenceToken,
    ];
    if (optionalValues.some((value) => value !== undefined) &&
        optionalValues.some((value) => value === undefined)) {
        throw new SandboxConfigurationError('agentRuntimeRoot, agentExecutable, inferenceSocketRoot and inferenceToken must be configured together');
    }
    const agentRuntimeRoot = options.agentRuntimeRoot === undefined
        ? undefined
        : existingDirectory(options.agentRuntimeRoot, 'agentRuntimeRoot');
    const inferenceSocketRoot = options.inferenceSocketRoot === undefined
        ? undefined
        : existingDirectory(options.inferenceSocketRoot, 'inferenceSocketRoot');
    if ((options.dependencyRoot === undefined) !== (options.dependencyTarget === undefined)) {
        throw new SandboxConfigurationError('dependencyRoot and dependencyTarget must be configured together');
    }
    const dependencyRoot = options.dependencyRoot === undefined
        ? undefined
        : existingDirectory(options.dependencyRoot, 'dependencyRoot');
    if (options.dependencyTarget !== undefined &&
        (!SAFE_MOUNT_TARGET.test(options.dependencyTarget) ||
            options.dependencyTarget.split('/').some((part) => part === '.' || part === '..'))) {
        throw new SandboxConfigurationError('dependencyTarget must be a safe relative path');
    }
    const roots = [
        workspace,
        runtimeRoot,
        profileRoot,
        agentRuntimeRoot,
        inferenceSocketRoot,
        dependencyRoot,
    ].filter((value) => value !== undefined);
    for (let leftIndex = 0; leftIndex < roots.length; leftIndex += 1) {
        for (let rightIndex = leftIndex + 1; rightIndex < roots.length; rightIndex += 1) {
            const left = roots[leftIndex];
            const right = roots[rightIndex];
            if (isWithin(left, right) || isWithin(right, left)) {
                throw new SandboxConfigurationError('sandbox mount roots must be disjoint');
            }
        }
    }
    const executable = resolve(runtimeRoot, options.runtimeExecutable);
    let executableRealPath;
    try {
        executableRealPath = realpathSync(executable);
        accessSync(executableRealPath, options.runtimeNodeEntrypoint === true ? constants.R_OK : constants.X_OK);
    }
    catch {
        throw new SandboxConfigurationError('runtimeExecutable must exist and be executable');
    }
    if (!isWithin(runtimeRoot, executableRealPath)) {
        throw new SandboxConfigurationError('runtimeExecutable must stay inside runtimeRoot');
    }
    let agentExecutable;
    if (agentRuntimeRoot !== undefined) {
        if (!SAFE_EXECUTABLE.test(options.agentExecutable)) {
            throw new SandboxConfigurationError('agentExecutable must be a safe relative path');
        }
        try {
            agentExecutable = realpathSync(resolve(agentRuntimeRoot, options.agentExecutable));
            accessSync(agentExecutable, constants.R_OK);
        }
        catch {
            throw new SandboxConfigurationError('agentExecutable must exist and be readable');
        }
        if (!isWithin(agentRuntimeRoot, agentExecutable)) {
            throw new SandboxConfigurationError('agentExecutable must stay inside agentRuntimeRoot');
        }
        if (!SAFE_TOKEN.test(options.inferenceToken)) {
            throw new SandboxConfigurationError('inferenceToken must be 64 lowercase hexadecimal characters');
        }
    }
    const args = options.args ?? [];
    if (args.length > 64 || args.some((value) => value.length > 4096 || value.includes('\0'))) {
        throw new SandboxConfigurationError('job arguments exceed the fixed protocol limits');
    }
    const requested = options.limits ?? DEFAULT_SANDBOX_LIMITS;
    const limits = {
        memoryBytes: boundedInteger(requested.memoryBytes, 128 * 1024 * 1024, 32 * 1024 * 1024 * 1024, 'memoryBytes'),
        cpuQuotaPercent: boundedInteger(requested.cpuQuotaPercent, 10, 800, 'cpuQuotaPercent'),
        tasksMax: boundedInteger(requested.tasksMax, 8, 512, 'tasksMax'),
        runtimeSeconds: boundedInteger(requested.runtimeSeconds, 1, 8 * 60 * 60, 'runtimeSeconds'),
        tmpBytes: boundedInteger(requested.tmpBytes, 16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 'tmpBytes'),
    };
    return {
        ...options,
        workspace,
        runtimeRoot,
        profileRoot,
        executable: executableRealPath,
        agentRuntimeRoot,
        agentExecutable,
        inferenceSocketRoot,
        dependencyRoot,
        dependencyTarget: options.dependencyTarget,
        args,
        limits,
    };
}
export function buildBubblewrapArguments(options) {
    const resolved = resolveOptions(options);
    const executableInSandbox = `/runtime/${relative(resolved.runtimeRoot, resolved.executable)}`;
    const command = resolved.runtimeNodeEntrypoint
        ? ['/usr/bin/node', executableInSandbox]
        : [executableInSandbox];
    const mounts = [];
    const environment = [];
    if (resolved.agentRuntimeRoot !== undefined &&
        resolved.agentExecutable !== undefined &&
        resolved.inferenceSocketRoot !== undefined) {
        mounts.push('--ro-bind', resolved.agentRuntimeRoot, '/agent-runtime', '--ro-bind', resolved.inferenceSocketRoot, '/inference');
        environment.push('--setenv', 'B70_AGENT_EXECUTABLE', `/agent-runtime/${relative(resolved.agentRuntimeRoot, resolved.agentExecutable)}`, '--setenv', 'B70_INFERENCE_TOKEN', resolved.inferenceToken);
    }
    return [
        '--unshare-all',
        '--unshare-user',
        '--die-with-parent',
        '--new-session',
        '--disable-userns',
        '--uid',
        '65534',
        '--gid',
        '65534',
        '--hostname',
        'b70-job',
        '--cap-drop',
        'ALL',
        '--ro-bind',
        '/usr',
        '/usr',
        '--symlink',
        'usr/bin',
        '/bin',
        '--symlink',
        'usr/lib',
        '/lib',
        '--symlink',
        'usr/lib',
        '/lib64',
        '--dev',
        '/dev',
        '--proc',
        '/proc',
        '--size',
        String(resolved.limits.tmpBytes),
        '--tmpfs',
        '/tmp',
        '--dir',
        '/home',
        '--dir',
        '/home/agent',
        resolved.workspaceWritable === false ? '--ro-bind' : '--bind',
        resolved.workspace,
        '/workspace',
        ...(resolved.dependencyRoot === undefined
            ? []
            : ['--ro-bind', resolved.dependencyRoot, `/workspace/${resolved.dependencyTarget}`]),
        '--ro-bind',
        resolved.runtimeRoot,
        '/runtime',
        resolved.writableProfile === true ? '--bind' : '--ro-bind',
        resolved.profileRoot,
        '/profile',
        ...mounts,
        '--chdir',
        '/workspace',
        '--clearenv',
        '--setenv',
        'HOME',
        '/home/agent',
        '--setenv',
        'PATH',
        '/usr/bin:/bin',
        '--setenv',
        'LANG',
        'C.UTF-8',
        '--setenv',
        'PI_OFFLINE',
        '1',
        '--setenv',
        'PI_CODING_AGENT_DIR',
        '/profile',
        '--setenv',
        'B70_RUN_MODE',
        resolved.runMode ?? 'implementation',
        ...environment,
        '--',
        ...command,
        ...resolved.args,
    ];
}
export function unitNameFor(runId) {
    if (!SAFE_RUN_ID.test(runId)) {
        throw new SandboxConfigurationError('runId must be 8–64 lowercase hexadecimal characters');
    }
    return `local-ai-b70-job-${runId}.scope`;
}
export function buildSystemdScopeArguments(options) {
    const resolved = resolveOptions(options);
    const unitName = unitNameFor(resolved.runId);
    return [
        '--user',
        '--scope',
        '--quiet',
        '--collect',
        `--unit=${unitName.slice(0, -'.scope'.length)}`,
        `--property=MemoryMax=${resolved.limits.memoryBytes}`,
        '--property=MemorySwapMax=0',
        `--property=TasksMax=${resolved.limits.tasksMax}`,
        `--property=CPUQuota=${resolved.limits.cpuQuotaPercent}%`,
        `--property=RuntimeMaxSec=${resolved.limits.runtimeSeconds}s`,
        '--property=OOMPolicy=kill',
        '--property=KillMode=control-group',
        BUBBLEWRAP,
        ...buildBubblewrapArguments(options),
    ];
}
/** Launch a job in a transient, non-restarting systemd scope and bwrap sandbox. */
export function launchSandboxedJob(options) {
    const unitName = unitNameFor(options.runId);
    const child = spawn(SYSTEMD_RUN, buildSystemdScopeArguments(options), {
        stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { child, unitName };
}
export function buildWorkspaceUtilityArguments(options) {
    if (!SAFE_RUN_ID.test(options.runId)) {
        throw new SandboxConfigurationError('runId must be 8–64 lowercase hexadecimal characters');
    }
    if (!SAFE_EXECUTABLE.test(options.runtimeExecutable)) {
        throw new SandboxConfigurationError('runtimeExecutable must be a safe relative path');
    }
    const workspace = existingDirectory(options.workspacePath, 'workspacePath');
    const runtimeRoot = existingDirectory(options.runtimeRoot, 'runtimeRoot');
    const sourcePath = options.sourcePath === undefined
        ? undefined
        : existingDirectory(options.sourcePath, 'sourcePath');
    const dependencyRoot = options.dependencyRoot === undefined
        ? undefined
        : existingDirectory(options.dependencyRoot, 'dependencyRoot');
    if ((dependencyRoot === undefined) !== (options.dependencyTarget === undefined)) {
        throw new SandboxConfigurationError('utility dependency mount is incomplete');
    }
    if (options.dependencyTarget !== undefined &&
        (!SAFE_MOUNT_TARGET.test(options.dependencyTarget) ||
            options.dependencyTarget.split('/').some((part) => part === '.' || part === '..'))) {
        throw new SandboxConfigurationError('dependencyTarget must be a safe relative path');
    }
    if (isWithin(workspace, runtimeRoot) || isWithin(runtimeRoot, workspace)) {
        throw new SandboxConfigurationError('utility mount roots must be disjoint');
    }
    if (sourcePath !== undefined &&
        [workspace, runtimeRoot].some((root) => isWithin(root, sourcePath) || isWithin(sourcePath, root))) {
        throw new SandboxConfigurationError('utility mount roots must be disjoint');
    }
    const executable = realpathSync(resolve(runtimeRoot, options.runtimeExecutable));
    if (!isWithin(runtimeRoot, executable)) {
        throw new SandboxConfigurationError('runtimeExecutable must stay inside runtimeRoot');
    }
    accessSync(executable, constants.R_OK);
    const args = options.args ?? [];
    if (args.length > 64 || args.some((value) => value.length > 16_384 || value.includes('\0'))) {
        throw new SandboxConfigurationError('utility arguments exceed the fixed protocol limits');
    }
    const requested = options.limits ?? {
        ...DEFAULT_SANDBOX_LIMITS,
        runtimeSeconds: 15 * 60,
        memoryBytes: 2 * 1024 * 1024 * 1024,
    };
    const limits = {
        memoryBytes: boundedInteger(requested.memoryBytes, 128 * 1024 * 1024, 32 * 1024 * 1024 * 1024, 'memoryBytes'),
        cpuQuotaPercent: boundedInteger(requested.cpuQuotaPercent, 10, 800, 'cpuQuotaPercent'),
        tasksMax: boundedInteger(requested.tasksMax, 8, 512, 'tasksMax'),
        runtimeSeconds: boundedInteger(requested.runtimeSeconds, 1, 8 * 60 * 60, 'runtimeSeconds'),
        tmpBytes: boundedInteger(requested.tmpBytes, 16 * 1024 * 1024, 4 * 1024 * 1024 * 1024, 'tmpBytes'),
    };
    const executableInSandbox = `/runtime/${relative(runtimeRoot, executable)}`;
    return [
        '--user',
        '--scope',
        '--quiet',
        '--collect',
        `--unit=${unitNameFor(options.runId).slice(0, -'.scope'.length)}`,
        `--property=MemoryMax=${limits.memoryBytes}`,
        '--property=MemorySwapMax=0',
        `--property=TasksMax=${limits.tasksMax}`,
        `--property=CPUQuota=${limits.cpuQuotaPercent}%`,
        `--property=RuntimeMaxSec=${limits.runtimeSeconds}s`,
        '--property=OOMPolicy=kill',
        '--property=KillMode=control-group',
        BUBBLEWRAP,
        '--unshare-all',
        '--unshare-user',
        '--die-with-parent',
        '--new-session',
        '--disable-userns',
        '--uid',
        '65534',
        '--gid',
        '65534',
        '--hostname',
        'b70-job',
        '--cap-drop',
        'ALL',
        '--ro-bind',
        '/usr',
        '/usr',
        '--symlink',
        'usr/bin',
        '/bin',
        '--symlink',
        'usr/lib',
        '/lib',
        '--symlink',
        'usr/lib',
        '/lib64',
        '--dev',
        '/dev',
        '--proc',
        '/proc',
        '--size',
        String(limits.tmpBytes),
        '--tmpfs',
        '/tmp',
        '--bind',
        workspace,
        '/workspace',
        ...(sourcePath === undefined ? [] : ['--ro-bind', sourcePath, '/result']),
        ...(dependencyRoot === undefined
            ? []
            : ['--ro-bind', dependencyRoot, `/workspace/${options.dependencyTarget}`]),
        '--ro-bind',
        runtimeRoot,
        '/runtime',
        '--chdir',
        '/workspace',
        '--clearenv',
        '--setenv',
        'HOME',
        '/tmp',
        '--setenv',
        'PATH',
        '/usr/bin:/bin',
        '--setenv',
        'LANG',
        'C.UTF-8',
        '--',
        '/usr/bin/node',
        executableInSandbox,
        ...args,
    ];
}
/** Run trusted fixed logic against repository content without host/network access. */
export function launchWorkspaceUtility(options) {
    const unitName = unitNameFor(options.runId);
    const child = spawn(SYSTEMD_RUN, buildWorkspaceUtilityArguments(options), {
        stdio: ['pipe', 'pipe', 'pipe'],
    });
    return { child, unitName };
}
/** Stop exactly one validated job scope. No arbitrary unit name is accepted. */
export async function stopSandboxedJob(runId) {
    const unitName = unitNameFor(runId);
    await new Promise((resolvePromise, reject) => {
        execFile(SYSTEMCTL, ['--user', 'kill', '--kill-whom=all', '--signal=SIGTERM', unitName], { timeout: 10_000 }, (error) => (error === null ? resolvePromise() : reject(error)));
    });
}
