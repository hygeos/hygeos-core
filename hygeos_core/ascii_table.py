import time
from typing import Literal


class ascii_table:
    
    """
    A class to represent a table for displaying data in a formatted way.
    """
    
    class style:
        """
        A class to define the style of a table using specific characters.
        """
        def __init__(self, 
            inner_vbar = True,
            inner_hbar = False,
            h_padding = 1,
            v_padding = 0,
            style: Literal["squared", "rounded", "simple", "double"] = "squared",
        ):
        
            styles = dict(
                squared = "─│┌┐└┘├┤┬┴┼",
                rounded = "─│╭╮╰╯├┤┬┴┼",
                simple  = "-|+++++++++",
                double  = "═║╔╗╚╝╠╣╦╩╬",
            )
            
            self.style = styles[style]
        
            self.inner_vbar = inner_vbar
            self.inner_hbar = inner_hbar
            self.h_padding = h_padding
            self.v_padding = v_padding
        
            # internal config
            self.h      = self.style[0]   # horizontal char
            self.v      = self.style[1]   # vertical char
            self.tl     = self.style[2]   # corner top left
            self.tr     = self.style[3]   # corner top right
            self.bl     = self.style[4]   # corner bot left
            self.br     = self.style[5]   # corner bot right
            self.ml     = self.style[6]   # mid left 
            self.mr     = self.style[7]   # mid right
            self.mt     = self.style[8]   # mid top
            self.mb     = self.style[9]   # mid bot
            self.cr     = self.style[10]  # cross
        
            self.live_print_target_time = 0.100 # s
    
    def __init__(self, df, style=style(), colors={}, sides={}, max_width: int|None=35):
        """
        Ascii table constructor
        df: source table, pd.DataFrame
        style: allows to override the styling of the table
        colors: is a dictionnary of hygeos_core.log.rgb colors like {column: rgb.green}
        
        Basic usage:
                ascii_table(res).print()
        """
        self.df = df
        self._style = style
        
        df_str = df.astype(str)
        self.columns: list[str] = df_str.columns
        self.spaces: list[int] = [
            max(len(str(c)), int(df_str.iloc[:, i].str.len().max()) if len(df_str) else 0)
            for i, c in enumerate(self.columns)
        ]
        
        # NOTE: could use interactive / dynamic ranges, more complex tough
        # self.terminal = shutil.get_terminal_size()
        
        # lowercase the columns keys
        colors = {k.lower(): v for k, v in colors.items()}
        sides = {k.lower(): v for k, v in sides.items()}
        
        
        for c in self.columns:
            c = str(c)
            if c.lower() not in colors: # no colors by default
                colors[c.lower()] = None
                
            if c.lower() not in sides: # left side by default
                sides[c.lower()] = "left"
        
        self.colors = [colors[str(i).lower()] for i in self.columns]
        self.sides = [sides[str(i).lower()] for i in self.columns]
        
        # limiting column max width
        self.max_width = max_width
        if max_width is not None:
            for i, v in enumerate(self.spaces):
                if v > max_width:
                    self.spaces[i] = max_width
        
        
        
    def print(self, live_print=False, no_color=False):
        s = self.to_string(live_print, no_color)
        print(s)
        
    
    def to_string(self, live_print=False, no_color=True):
        """
        Generates an ascii representation of the dataset provided in the constructor.
        """
        
        # NOTE: does not (yet) handle line reduction with ellipsis
            # ellipsis_char = "‥" # "…" # two dots variants seems more readable 
            # width_footprint = sum(self.widths) + 3*len(self.widths) + 1
            # width_overshoot = width_footprint - self.terminal.columns
    
        # config
        ivb = self._style.inner_vbar
        ihb = self._style.inner_hbar
        hp  = self._style.h_padding
        vp  = self._style.v_padding
        
        # style characters
        h   = self._style.h  # horizontal
        v   = self._style.v  # vertical 
        tl  = self._style.tl # top left corner            
        tr  = self._style.tr # top right corner
        bl  = self._style.bl # bottom left
        br  = self._style.br # bottom right
        mr  = self._style.mr # mid right              
        mt  = self._style.mt # mid top         
        ml  = self._style.ml # mid left           
        mb  = self._style.mb # mid bottom         
        cr  = self._style.cr # cross
    
        max_width = self.max_width
    
        spaces = self.spaces.copy()
        colors = self.colors.copy()
        sides = self.sides.copy()
    
        # Precompute common strings
        hhp = h * hp
        shp = ' ' * hp
        h_line = h * (hp * 2 + 1)  # For cases without inner bars
        
        def _line_separator(pos):
            """
            Generates an ascii separator line for the table based on position.
            """
            
            if pos == "mid":
                
                ivbc = cr if ivb else h       # cross if inner vertical bars, line otherwise
                sep = hhp + ivbc + hhp      # construct mid separator (not on the outer side)
                if not ivb: sep = hhp        # if no vertical bar, reduce padding
                
                mids = [h*w for w in spaces]
                line = ml + hhp + sep.join(mids) + hhp + mr
                return line
            
            if pos == "top":
                
                ivbc = mt if ivb else h
                sep = hhp + ivbc + hhp
                if not ivb: sep = hhp
                
                mids = [h*w for w in spaces]
                line = tl + hhp + sep.join(mids) + hhp + tr
                return line
            
            if pos == "bot":
                
                ivbc = mb if ivb else h
                sep = hhp + ivbc + hhp
                if not ivb: sep = hhp
                
                mids = [h*w for w in spaces]
                line = bl + hhp + sep.join(mids) + hhp + br
                return line
            
        
        def _line_padding():
            """
            Generates an ascii padding line for the table.
            """

            # same layout as _line_content, with empty cells
            sep = shp + v + shp
            if not ivb: sep = shp
        
            mids = [" "*w for w in spaces]
            line = v + shp + sep.join(mids) + shp + v
            return line
        
        
        def _color_str(s, i):
            c = colors[i]

            if c is not None:
                return c(s)
            return s
        
        def _format_str(s):
            
            l = max_width
            if len(s) > l:
                s = s[:l-1] + "‥"
            return s

                
        # disable for performance reason
        if not max_width:
            _format_str = lambda x: x
        
        # disable colors
        if no_color:
            _color_str = lambda x, y: x
        
        def _line_content(line):
            """
            Generates an ascii content line for the table.
            """
            
            ivbc = v if ivb else h
            sep = shp + ivbc + shp
            if not ivb: sep = shp
        
            def align_text(text, width, side):
                return text.ljust(width) if side == "left" else text.rjust(width)

            mids = [
                _color_str(align_text(_format_str(str(w)), spaces[i], sides[i]), i) 
                for i, w in enumerate(line)
]
            line = v + shp + sep.join(mids) + shp + v
            return line
    
    
        """
        Main routine using the previous subroutines
        Iterates over each lines of the dataframe and builds the ascii representation
        """
        
        entries = self.df[self.columns].values
        lines = []
        
        headers = [str(s).capitalize() for s in self.columns]
        
        
        itr = self._style.live_print_target_time / max(len(self.df), 1)
        itr = min(itr, 0.015) 
        fn = lines.append if not live_print else lambda x: (time.sleep(itr), print(x, flush=True))
        
        fn("")
        fn(_line_separator("top" ))
        fn(_line_content(headers))
        fn(_line_separator("mid" ))
        for j, e in enumerate(entries):
            
            for i in range(vp): fn(_line_padding())
            fn(_line_content(e))
            for i in range(vp): fn(_line_padding())
            if ihb:
                fn(_line_separator("mid"))
                
        fn(_line_separator("bot"))
        fn("")
        
        if not live_print: 
            message = "\n".join(lines)
            return message