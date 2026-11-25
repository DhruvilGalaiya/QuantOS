import { LogIn, LogOut, Moon } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandShortcut } from "@/components/ui/command";
import { SignInDialog } from "@/components/SignIn";
import { useOwner, useSignOut } from "@/lib/auth";
import { NAV } from "@/lib/nav";
import { useTheme } from "@/lib/theme";

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (v: boolean) => void }) {
  const navigate = useNavigate();
  const toggleTheme = useTheme((s) => s.toggle);
  const { data: me } = useOwner();
  const signOut = useSignOut();
  const [signInOpen, setSignInOpen] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  const run = (fn: () => void) => {
    onOpenChange(false);
    fn();
  };

  return (
    <>
      <SignInDialog open={signInOpen} onOpenChange={setSignInOpen} />
      <CommandDialog open={open} onOpenChange={onOpenChange} title="Search QuantOS" description="Jump to a page or run a command">
        <CommandInput placeholder="Go to a page or run a command" />
        <CommandList>
          <CommandEmpty>Nothing matches that. Try a page name such as “risk”.</CommandEmpty>
          {NAV.map((g) => (
            <CommandGroup key={g.group} heading={g.group}>
              {g.items.map((it) => (
                <CommandItem key={it.to} value={`${it.label} ${it.blurb}`} onSelect={() => run(() => navigate(it.to))}>
                  <it.icon className="size-4" />
                  <span className="shrink-0">{it.label}</span>
                  <span className="hidden min-w-0 truncate text-muted-foreground sm:block">{it.blurb}</span>
                  <CommandShortcut className="shrink-0 whitespace-nowrap pl-2">g {it.key}</CommandShortcut>
                </CommandItem>
              ))}
            </CommandGroup>
          ))}
          <CommandGroup heading="Settings">
            <CommandItem value="Switch theme light dark" onSelect={() => run(toggleTheme)}>
              <Moon className="size-4" />
              <span>Switch light / dark theme</span>
            </CommandItem>
            {me?.enabled && !me.owner && (
              <CommandItem value="Sign in owner live prices" onSelect={() => run(() => setSignInOpen(true))}>
                <LogIn className="size-4" />
                <span>Sign in for live prices</span>
              </CommandItem>
            )}
            {me?.owner && (
              <CommandItem value="Sign out owner" onSelect={() => run(() => signOut.mutate())}>
                <LogOut className="size-4" />
                <span>Sign out</span>
              </CommandItem>
            )}
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  );
}
