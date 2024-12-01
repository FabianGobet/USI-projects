from typing import List, Union
import matplotlib.pyplot as plt

class Template:
    def __init__(self, values: list, args: List[Union[str, int, float]]):
        self.dict = dict(zip(args, values))
        
    def __str__(self):
        return ', '.join([f'{key}: {value}' for key, value in self.dict.items()])
    
    def get_values(self):
        return self.dict.values()
    
    def get_attribute(self, attr):
        return self.dict[attr]
    
    def __len__(self):
        return len(self.dict)
    


class Archive:
    def __init__(self, args, query_length: int, templates: List[Template] = []):
        assert isinstance(args, list) and len(args)>0, 'Invalid arguments.'
        assert len(templates) == 0 or all(isinstance(template, Template) and len(template) == len(args) for template in templates), 'Invalid templates.'
        assert isinstance(query_length, int) and query_length > 0, 'Invalid query length.'
        self.args = args
        self.templates = templates 
        self.query_length = query_length
    
    def add_template(self, template: Template):
        self.templates.append(template)
    
    def order_by_parameters(self, reverse=False, func = None, *parameters):
        assert self.templates is not None, 'No templates to order.'
        if func is not None:
            self.templates.sort(key = lambda template: func([template.get_attribute(parameter) for parameter in parameters]), reverse = reverse)
        else:
            self.templates.sort(key = lambda template: [template.get_attribute(parameter) for parameter in parameters], reverse = reverse)

    def __str__(self) -> str:
        r = ''
        for template in self.templates:
            r += str(template)+'\n'
        return r
    
    def checks_filters(self, filters: dict, template: Template):
        for key, value in filters.items():
            item = template.get_attribute(key)
            if isinstance(value, tuple):
                to_include, values = value
                assertion1 = to_include and not any([v in item for v in values])
                assertion2 = any([v in item for v in values]) and not to_include
                if assertion1 or assertion2:
                    print(f'{template.get_attribute("No")} Ruled out by {key}:{item} filter.')
                    return False
            elif not value[0] <= item <= value[1]:
                print(f'{template.get_attribute("No")} Ruled out by {key}:{item} filter.')
                return False
        return True
    
    def save_to_csv(self, filename: str,filters: dict = None,  saving_args=[]):
        hit_list = []
        with open(filename, 'w') as f:
            f.write(f'{",".join(self.args if len(saving_args) == 0 else saving_args)}' + '\n')
            for template in self.templates:
                if filters is None or self.checks_filters(filters, template):
                    hit_list.append(template.get_attribute('Hit'))
                    if len(saving_args) == 0:
                        f.write(','.join(map(str, template.get_values())) + '\n')
                    else:
                        f.write(','.join(map(str, [template.get_attribute(arg) for arg in saving_args])) + '\n')
        return hit_list
                
    def get_args(self):
        return self.args
    
    def get_query_length(self):
        return self.query_length
    
    def get_attribute(self, attr):
        return [template.get_attribute(attr) for template in self.templates]
    
    def __len__(self):
        return len(self.templates)
    
    def get_frequency_coverage_for_hits(self, hit_list: List[str]) -> List[int]:
        frequencies = [0 for i in range(self.query_length)]
        for t in self.templates:
            if t.get_attribute('Hit').split('_')[0] in hit_list:
                a,b = t.get_attribute('Query HMM').strip('"').split(', ')
                a = int(a.removeprefix('['))-1
                b = int(b.removesuffix(']'))-1
                for i in range(a,b+1):
                    frequencies[i] += 1
        return frequencies
    
    def _red_green_color_map(self, frequencies: list) -> list:
        colors = []
        max_value = max(frequencies)
        for v in frequencies:
            colors.append((1-v/max_value, v/max_value, 0))
        return colors
    
    def get_average_frequency(self, frequencies: list) -> float:
        return sum(frequencies)/len(frequencies)
    
    def plot_frequency_coverage(self, frequencies: list, num_x_ticks = 7) -> float:
        colors = self._red_green_color_map(frequencies)
        shifted = [i+1 for i in range(len(frequencies))]
        plt.bar(shifted, frequencies, color=colors)
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title('Frequency coverage for hits')
        x_ticks = shifted[::int(len(shifted)/num_x_ticks)]
        x_ticks = x_ticks[0:len(x_ticks)-1] + [shifted[-1]]
        plt.xticks(x_ticks)
        plt.show()
        
    def get_templates_with_trimmed_hits(self, hit_list: List[str]) -> dict:
        dict_hits = {}
        for t in self.templates:
            hit_trimmed = t.get_attribute('Hit').split('_')[0]
            if hit_trimmed in hit_list:
                dict_hits[hit_trimmed] = t
        return dict_hits