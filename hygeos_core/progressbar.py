import sys
import time
from typing import Literal

from hygeos_core import log
from hygeos_core import env
import shutil

# Add IPython support for notebooks
try:
    from IPython.display import clear_output
    from IPython import get_ipython
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False



class _utils:
    
    def get_terminal_width():
        size = shutil.get_terminal_size((80, 20))
        return size.columns
    
    def inside_notebook():
        """Detect if we're running in a Jupyter notebook"""
        if not HAS_IPYTHON:
            return False
        try:
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'  # Jupyter notebook
        except (NameError, AttributeError):
            return False
        

class msg_stack:

    stack = []
        
    def get(obj):
        for item in msg_stack.stack:
            if item == obj: return item
            # else: print(item.object, obj)
        raise KeyError("Object not found in stack")

    def add(obj):
        if not obj in msg_stack.stack:
            msg_stack.stack.append(obj)
        else:
            raise KeyError("Object already in stack")

    def clean_above(obj: 'progressbar'):
        # remove the objects above from stack, excluding the object itself
        
        obj = msg_stack.get(obj)
        if obj in msg_stack.stack:
            index = msg_stack.stack.index(obj)
            
            # intercept case where it's the last item
            if index == len(msg_stack.stack) - 1:
                return
            
            msg_stack.reset_screen()
            msg_stack.stack = msg_stack.stack[:index + 1]
        else:
            # should not happen
            raise KeyError("Object not found in stack")
    
    
    def reset_screen():
        
        size = len(msg_stack.stack)
        
        msg = (" " * _utils.get_terminal_width() + "\n") * size
        log.disp(msg, end='')
        msg_stack.reset_cursor(size)
    
    
    def info(*args, **kwargs):
        
        msg_stack.reset_screen()
        log.info(*args, **kwargs)
        msg_stack.update_and_print_stack()
        
        
        
    def update_and_print_stack():
        
        if len(msg_stack.stack) == 1 and msg_stack.stack[0].finished: 
            msg_stack.reset_screen()
            msg_stack.stack = []
            return
        
        meta_msg = ""
        for obj in msg_stack.stack:
            msg = obj.update()
            meta_msg += msg + "\n"
        
        log.disp(meta_msg, end='')
        size = meta_msg.count('\n')
        
        msg_stack.reset_cursor(size)
        
    
    def reset_cursor(size: int):
        if size <= 0: return
        
        # Move the cursor up to the start of the stack
        print(f"\r\033[{size}A", end='', flush=True)
        
            
            
class progressbar:
    
    pbar_length = 30
    
    def __init__(self, iterable, prefix, nth=1):
        self.iterable = iterable
        self.iterator = None
        
        self.length = len(iterable)
        self.current = 0
        
        self.prefix = prefix
        self.nth = nth
        
        self.frame_counter = 0
        self.message = ""
        
        self.start_time = time.time()
        self.finished = False
        
        if self.length > 0 and self.nth > self.length:
            raise ValueError("nth cannot be greater than the length of the iterable")
        
        if not self in msg_stack.stack:
            msg_stack.add(self)
    
    
    def __iter__(self):
        self.iterator = iter(self.iterable)
        self.current = 0
        
        # allow iterating the same bar again after it finished
        if self.finished:
            self.finished = False
            self.start_time = time.time()
        if self not in msg_stack.stack:
            msg_stack.add(self)
        
        # generator wrapping __next__, so that the bar is removed from the
        # stack when the loop is exited early (break, exception)
        try:
            while True:
                try:
                    n = self.__next__()
                except StopIteration:
                    return
                yield n
        finally:
            if not self.finished and self in msg_stack.stack:
                msg_stack.reset_screen()
                index = msg_stack.stack.index(self)
                msg_stack.stack = msg_stack.stack[:index]
                if msg_stack.stack:
                    msg_stack.update_and_print_stack()
        
    def update(self):
        if self.finished: return log.rgb.green(self.message)
        self.message = self.ascii_pbar(self.pbar_length)        
        return log.rgb.orange(self.message)
            
    def finish(self):
        self.finished = True
        self.message = self.ascii_pbar(self.pbar_length)
    
    def __next__(self):
        try:
            
            if self.nth == 1: 
                display = True
            else:
                nth   = bool((self.current+1) % self.nth == 0)
                first = bool(self.current+1 == 1)
                last  = bool(self.current+1 == self.length)
                display = nth or first or last 
                
            if display:
                msg_stack.clean_above(self)
                
                n = next(self.iterator)
                self.current += 1
                self.frame_counter += 1
                
                msg_stack.update_and_print_stack()
                
            else:
                n = next(self.iterator)
                self.current += 1
                
            return n
            
        except StopIteration:
            
            self.finish()
            msg_stack.update_and_print_stack()
            
            raise
    
    def ascii_loading(self, fps: int = 10, style: Literal["dots", "vbar", "moon", "earth"] = "dots"):
        
        f1 = "√"
        if self.finished: return f1
        
        styles = dict(
            # cdots = "⠁ |⠉ |⠉⠁|⠈⠉| ⠙| ⠸| ⢰| ⣠|⢀⣀|⣀⡀|⣄ |⡆ |⠇ |⠃ ",
            dots = "⠋|⠉|⠙|⠸|⢰|⣠|⣄|⠇",
            vbar = "▁|▂|▃|▄|▅|▆|▇|█",
            moon = "🌘|🌗|🌖|🌕|🌔|🌓|🌒|🌑",
            earth = "🌍|🌎|🌏",
        )
        
        s = styles.get(style)
        c = s.split("|")
        
        if style == "vpbar":
            ratio = ((self.current -1 + int(self.finished)) / self.length)
            index = ratio * len(c)
            
            # Escape code to change background color to gray
            bgc = "\033[48;2;100;100;100m"
            return bgc + c[int(index)] + str(log.rgb.default)
        
        # Calculate frame index based on current time and desired FPS
        current_time = time.time()
        fps = len(c)
        frame_index = int(current_time * fps) % len(c)
        
        return c[frame_index]
    
    def get_depth(self):
        """
        Get the depth of the progress bar in the stack (ignoring finished bars)
        """
        depth = 0
        for item in msg_stack.stack:
            if item == self: return depth
            if not item.finished: depth += 1
        return depth
        

    def ascii_pbar(self, nchars: int, fmt: str = None, 
    bar_style: Literal[
        "square_dot", "square_void", 
        "block_void", "block_dot", 
        "hash_dash", "hash_void", 
        "equal_void"] = None, 
    border_style: Literal["brackets", "pipes", "none"] = None,
    icon_style: Literal["dots", "vbar", "moon", "earth"] = None
    ):
        """
        Generate an ASCII progress bar string.
        - nchars: Length of the progress bar in characters.
        - bar_style: any of:
            - square_dot  : [■■■···]
            - square_void : [■■■   ]
            - block_void  : [███░  ]
            - block_dot   : [███···]
            - hash_dash   : [###---]
            - hash_void   : [###   ]
            - equal_void  : [===   ]
        - border_style: any of:
            - brackets : [bar]
            - pipes    : |bar|
            - none     : bar
        - loading_style: any of:
            - dots  : ⣠
            - vbar  : ▁▂▃▄▅▆▇█
            - moon  : 🌗
            - earth : 🌍
        - fmt can contain:
            - %icon : loading animation
            - %pct     : percentage completed
            - %bar     : the progress bar itself
            - %itr     : current iteration / total iterations
            - %time    : elapsed time and estimated remaining time
            
        - if any of the style parameters is None, it will try to get it from the env var HYGEOS_PBAR_STYLE
          which is a string like "fmt|bar_style|border_style|icon_style"
        """
        
        env_style = env.getvar("HYGEOS_PBAR_STYLE", default="") # like: "square_dot|brackets|dots" or "square_void||dots"
        
        l: list = env_style.split("|")
        # fill list to 3 elements
        parts = l + [""] * (4 - len(l))
        parts = [l if l != "" else None for l in parts ]
        env_fmt, env_bar_style, env_border_style, env_loading_style = parts
    
        env_fmt           = parts[0] if parts[0] else None
        env_bar_style     = parts[1] if parts[1] else None
        env_border_style  = parts[2] if parts[2] else None
        env_icon_style    = parts[3] if parts[3] else None
        
        default_fmt = "%itr %icon %pct %bar %time"
        default_bar_style = "square_dot"
        default_border_style = "none"
        default_icon_style = "dots"
        
        # Apply priority: params > env > default
        final_fmt = fmt or env_fmt or default_fmt
        final_bar_style = bar_style or env_bar_style or default_bar_style
        final_border_style = border_style or env_border_style or default_border_style
        final_icon_style = icon_style or env_icon_style or default_icon_style
    
        
        styles = dict(
            square_dot  = "■·",
            square_void = "■ ",
            block_void  = "█░",
            block_dot   = "█·",
            hash_dash   = "#-",
            hash_void   = "# ",
            equal_void  = "= ",
        )
        
        border_styles = dict(
            brackets = "[]",
            pipes  = "||",
            none   = "",
        )
        
        s = styles.get(final_bar_style)
        b = border_styles.get(final_border_style)
        
        char, void = s[0], s[1]
        
        itr = self.current
        total = self.length
        
        ratio = ((itr -1 + int(self.finished)) / total) if total else float(self.finished)
        filled_length = int(nchars * ratio)
        
        bar = char * filled_length + void * (nchars - filled_length)
        
        if b:
            b1, b2 = b[0], b[1]
            bar = f"{b1}{bar}{b2}"
            
        
        pct = f"{ratio*100:.0f}%".rjust(3)
        # format counter with paded itr from total like 001/999
        nchar_total = len(str(total))
        padded_itr = str(itr).rjust(nchar_total, ' ')
        itr = f"{padded_itr}/{total}"
        
        etime = time.time() - self.start_time
        eta = "?" if ratio <= 0 else (etime * 1 / ratio) - etime
        
        def format_time(t):
            if t == "?": return t
            if t < 60:
                return f"{t:.0f}s"
            elif t < 3600:
                m = t // 60
                s = t % 60
                return f"{m:.0f}m {s:.0f}s"
            else:
                h = t // 3600
                m = (t % 3600) // 60
                s = t % 60
                
                parts = []
                if h > 0:
                    parts.append(f"{h:.0f}h")
                    
                if m > 0:
                    parts.append(f"{m:02.0f}m")
                    
                if s > 0 and h == 0: # only show seconds if no hours
                    parts.append(f"{s:02.0f}s")
                    
                return "".join(parts)
        
        
        stime = f"took {format_time(etime)}" if self.finished else f"{format_time(eta)} remaining"
        
        icon = self.ascii_loading(style=final_icon_style)
        
        color = log.rgb.orange if not self.finished else log.rgb.green
        
        # apply fmt
        fmt = final_fmt
        msg = fmt.replace("%icon",  color(icon))
        msg = msg.replace("%bar",   color(bar))
        msg = msg.replace("%pct",   color(pct))
        msg = msg.replace("%itr",   color(itr))
        msg = msg.replace("%time",  color(stime))
        
        prefix = self.prefix.strip()
        if not prefix.endswith(" "): prefix += " "
        
        depth = self.get_depth()
        nesting = "    " * (depth- 1) + "└── " * bool(depth > 0)
        
        
        message = f"{log.rgb.default(nesting)}{log.rgb.default(prefix)}{msg}".ljust(_utils.get_terminal_width())
        
        return message
