#!/bin/bash
# PreToolUse hook: hard-blocks raw Bash calls to known-verbose commands
# (dotnet build/test) and redirects to the compress_command_output tool on
# the local-compress MCP server, so their output never enters context
# uncompressed.
#
# This is deliberately a hard deny, not a soft nudge -- consistent with the
# rest of this project (fail loud, don't silently degrade). The real cost:
# if LM Studio isn't running, these commands become unusable via Bash until
# either LM Studio is started or this hook is temporarily disabled. The
# reason message below tells Claude to check with list_local_models and
# surface that to you rather than getting stuck silently.
#
# Registered via .mcp.json-style settings.json (see settings.json.template)
# with one hook handler per `if` pattern -- the `if` field only holds one
# permission rule, so bare-command and with-args forms need separate entries
# pointing at this same script.

input=$(cat)
command=$(echo "$input" | jq -r '.tool_input.command // empty')

if [ -z "$command" ]; then
  # Not a recognizable Bash command -- don't block, let it proceed.
  exit 0
fi

jq -n --arg cmd "$command" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("This command typically produces large, mostly-mechanical output. Use the compress_command_output tool (local-compress MCP server) with command=\"" + $cmd + "\" instead of running it via Bash directly -- it runs the command server-side and only the compressed summary enters context. If list_local_models shows LM Studio is unreachable or no model is loaded, tell the user and ask whether to bypass this hook and run the command via Bash anyway.")
  }
}'
